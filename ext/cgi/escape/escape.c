#include "ruby.h"
#include "ruby/encoding.h"
#include "ruby/util.h"

#ifndef HAVE_RB_LEN_T
typedef long rb_len_t;
# define RB_LEN_MAX LONG_MAX
#endif

#define lower_hexdigits (ruby_hexdigits+0)
#define upper_hexdigits (ruby_hexdigits+16)
#define char_to_number(c) ruby_digit36_to_number_table[(unsigned char)(c)]

static VALUE rb_cCGI, rb_mEscape, rb_mEscapeExt;
static ID id_accept_charset;

#define HTML_ESCAPE_MAX_LEN 6

static const struct {
    uint8_t len;
    char str[HTML_ESCAPE_MAX_LEN+1];
} html_escape_table[UCHAR_MAX+1] = {
#define HTML_ESCAPE(c, str) [c] = {rb_strlen_lit(str), str}
    HTML_ESCAPE('\'', "&#39;"),
    HTML_ESCAPE('&', "&amp;"),
    HTML_ESCAPE('"', "&quot;"),
    HTML_ESCAPE('<', "&lt;"),
    HTML_ESCAPE('>', "&gt;"),
#undef HTML_ESCAPE
};

static inline void
preserve_original_state(VALUE orig, VALUE dest)
{
    rb_enc_associate(dest, rb_enc_get(orig));
}

static inline rb_len_t
escaped_length(VALUE str)
{
    const rb_len_t len = RSTRING_LEN(str);
    if (len >= RB_LEN_MAX / HTML_ESCAPE_MAX_LEN) {
        ruby_malloc_size_overflow(len, HTML_ESCAPE_MAX_LEN);
    }
    return len * HTML_ESCAPE_MAX_LEN;
}

/*
 * CGI.escapeHTML always runs a byte-at-a-time table loop over the
 * whole string into a scratch buffer up to 6x the source size, and
 * throws that buffer away to duplicate the source whenever no
 * character needed escaping.  Plain-text payloads (the common case
 * for large strings) therefore pay a full scalar copy for a plain
 * dup.
 *
 * For long strings, probe for the five escape characters in a single
 * SIMD pass first and duplicate the source directly when none is
 * present, the way php-src's htmlspecialchars() fast path does.  The
 * probe touches each byte exactly once, so the cost when the string
 * does contain escape characters stays well below the table loop it
 * precedes.
 */
#define HTML_ESCAPE_MEMCHR_MIN_LEN 128

/* Returns true if [p, p+len) contains an HTML escape character. */
static int
html_escape_scan_plain(const unsigned char *p, size_t len)
{
    static const unsigned char needles[5] = { '\'', '&', '"', '<', '>' };
    size_t i, j;
    for (i = 0; i < len; i++) {
        for (j = 0; j < 5; j++) {
            if (p[i] == needles[j]) return 1;
        }
    }
    return 0;
}

#if defined(__SSE2__)
# include <emmintrin.h>
#endif

#if defined(__AVX2__)
# include <immintrin.h>
static int
html_escape_scan(const unsigned char *p, size_t len)
{
    const __m256i sq = _mm256_set1_epi8((char)'\'');
    const __m256i am = _mm256_set1_epi8((char)'&');
    const __m256i qu = _mm256_set1_epi8((char)'"');
    const __m256i lt = _mm256_set1_epi8((char)'<');
    const __m256i gt = _mm256_set1_epi8((char)'>');
    if (len < 32) return html_escape_scan_plain(p, len);
    while (len >= 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)p);
        __m256i m = _mm256_or_si256
            (_mm256_or_si256(_mm256_cmpeq_epi8(v, sq), _mm256_cmpeq_epi8(v, am)),
             _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, qu),
                                             _mm256_cmpeq_epi8(v, lt)),
                             _mm256_cmpeq_epi8(v, gt)));
        if (!_mm256_testz_si256(m, m)) return 1;
        p += 32;
        len -= 32;
    }
    return html_escape_scan_plain(p, len); /* short tail, at most 31 bytes */
}
#elif defined(__SSE2__)
html_escape_scan(const unsigned char *p, size_t len)
{
    const __m128i sq = _mm_set1_epi8((int)'\'');
    const __m128i am = _mm_set1_epi8((int)'&');
    const __m128i qu = _mm_set1_epi8((int)'"');
    const __m128i lt = _mm_set1_epi8((int)'<');
    const __m128i gt = _mm_set1_epi8((int)'>');
    if (len < 16) return html_escape_scan_plain(p, len);
    while (len >= 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)p);
        __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(v, sq), _mm_cmpeq_epi8(v, am)),
                                 _mm_or_si128(_mm_cmpeq_epi8(v, qu),
                                              _mm_cmpeq_epi8(v, lt)));
        m = _mm_or_si128(m, _mm_cmpeq_epi8(v, gt));
        if (_mm_movemask_epi8(m)) return 1;
        p += 16;
        len -= 16;
    }
    return html_escape_scan_plain(p, len); /* short tail, at most 15 bytes */
}
#elif defined(__ARM_NEON)
# include <arm_neon.h>
static int
html_escape_scan(const unsigned char *p, size_t len)
{
    const uint8x16_t s0 = vdupq_n_u8((uint8_t)'\'');
    const uint8x16_t s1 = vdupq_n_u8((uint8_t)'&');
    const uint8x16_t s2 = vdupq_n_u8((uint8_t)'"');
    const uint8x16_t s3 = vdupq_n_u8((uint8_t)'<');
    const uint8x16_t s4 = vdupq_n_u8((uint8_t)'>');
    if (len < 16) return html_escape_scan_plain(p, len);
    while (len >= 16) {
        uint8x16_t v = vld1q_u8(p);
        uint8x16_t m = vorrq_u8(vorrq_u8(vceqq_u8(v, s0), vceqq_u8(v, s1)),
                                vorrq_u8(vorrq_u8(vceqq_u8(v, s2), vceqq_u8(v, s3)),
                                         vceqq_u8(v, s4)));
        uint64x2_t u = vreinterpretq_u64_u8(m);
        if (vgetq_lane_u64(u, 0) | vgetq_lane_u64(u, 1)) return 1;
        p += 16;
        len -= 16;
    }
    return html_escape_scan_plain(p, len);
}
#else
# define html_escape_scan html_escape_scan_plain
#endif

static VALUE
optimized_escape_html(VALUE str)
{
    VALUE escaped;
    VALUE vbuf;
    char *buf;
    const char *cstr = RSTRING_PTR(str);
    const char *end = cstr + RSTRING_LEN(str);

    if (end - cstr >= HTML_ESCAPE_MEMCHR_MIN_LEN) {
        if (!html_escape_scan((const unsigned char *)cstr, (size_t)(end - cstr)))
            return rb_str_dup(str);
    }

    buf = ALLOCV_N(char, vbuf, escaped_length(str));
    char *dest = buf;
    while (cstr < end) {
        const unsigned char c = *cstr++;
        uint8_t len = html_escape_table[c].len;
        if (len) {
            memcpy(dest, html_escape_table[c].str, len);
            dest += len;
        }
        else {
            *dest++ = c;
        }
    }

    if (RSTRING_LEN(str) < (dest - buf)) {
        escaped = rb_str_new(buf, dest - buf);
        preserve_original_state(str, escaped);
    }
    else {
        escaped = rb_str_dup(str);
    }
    ALLOCV_END(vbuf);
    return escaped;
}

static VALUE
optimized_unescape_html(VALUE str)
{
    enum {UNICODE_MAX = 0x10ffff};
    rb_encoding *enc = rb_enc_get(str);
    unsigned long charlimit = (strcasecmp(rb_enc_name(enc), "UTF-8") == 0 ? UNICODE_MAX :
                               strcasecmp(rb_enc_name(enc), "ISO-8859-1") == 0 ? 256 :
                               128);
    rb_len_t i, j, len, beg = 0;
    size_t clen, plen;
    int overflow;
    const char *cstr;
    char buf[6];
    VALUE dest = 0;

    len  = RSTRING_LEN(str);
    cstr = RSTRING_PTR(str);

    for (i = 0; i < len; i++) {
        unsigned long cc;
        char c = cstr[i];
        if (c != '&') continue;
        plen = i - beg;
        if (++i >= len) break;
        c = (unsigned char)cstr[i];
        j = i;
#define MATCH(s) (len - i >= (int)rb_strlen_lit(s) && \
                  memcmp(&cstr[i], s, rb_strlen_lit(s)) == 0 && \
                  (i += rb_strlen_lit(s) - 1, 1))
        switch (c) {
          case 'a':
            ++i;
            if (MATCH("pos;")) {
                c = '\'';
            }
            else if (MATCH("mp;")) {
                c = '&';
            }
            else {
                i = j;
                continue;
            }
            break;
          case 'q':
            ++i;
            if (MATCH("uot;")) {
                c = '"';
            }
            else {
                i = j;
                continue;
            }
            break;
          case 'g':
            ++i;
            if (MATCH("t;")) {
                c = '>';
            }
            else {
                i = j;
                continue;
            }
            break;
          case 'l':
            ++i;
            if (MATCH("t;")) {
                c = '<';
            }
            else {
                i = j;
                continue;
            }
            break;
          case '#':
            if (len - ++i >= 2 && ISDIGIT(cstr[i])) {
                cc = ruby_scan_digits(&cstr[i], len-i, 10, &clen, &overflow);
            }
            else if ((cstr[i] == 'x' || cstr[i] == 'X') && len - ++i >= 2 && ISXDIGIT(cstr[i])) {
                cc = ruby_scan_digits(&cstr[i], len-i, 16, &clen, &overflow);
            }
            else {
                i = j;
                continue;
            }
            i += clen;
            if (overflow || cc >= charlimit || cstr[i] != ';') {
                i = j;
                continue;
            }
            if (!dest) {
                dest = rb_str_buf_new(len);
            }
            rb_str_cat(dest, cstr + beg, plen);
            if (charlimit > 256) {
                rb_str_cat(dest, buf, rb_enc_mbcput((OnigCodePoint)cc, buf, enc));
            }
            else {
                c = (unsigned char)cc;
                rb_str_cat(dest, &c, 1);
            }
            beg = i + 1;
            continue;
          default:
            --i;
            continue;
        }
        if (!dest) {
            dest = rb_str_buf_new(len);
        }
        rb_str_cat(dest, cstr + beg, plen);
        rb_str_cat(dest, &c, 1);
        beg = i + 1;
    }

    if (dest) {
        rb_str_cat(dest, cstr + beg, len - beg);
        preserve_original_state(str, dest);
        return dest;
    }
    else {
        return rb_str_dup(str);
    }
}

static unsigned char
url_unreserved_char(unsigned char c)
{
    switch (c) {
      case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
      case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g': case 'h': case 'i': case 'j':
      case 'k': case 'l': case 'm': case 'n': case 'o': case 'p': case 'q': case 'r': case 's': case 't':
      case 'u': case 'v': case 'w': case 'x': case 'y': case 'z':
      case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G': case 'H': case 'I': case 'J':
      case 'K': case 'L': case 'M': case 'N': case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T':
      case 'U': case 'V': case 'W': case 'X': case 'Y': case 'Z':
      case '-': case '.': case '_': case '~':
        return 1;
      default:
        break;
    }
    return 0;
}

static VALUE
optimized_escape(VALUE str, int plus_escape)
{
    rb_len_t i, len, beg = 0;
    VALUE dest = 0;
    const char *cstr;
    char buf[4] = {'%'};

    len  = RSTRING_LEN(str);
    cstr = RSTRING_PTR(str);

    for (i = 0; i < len; ++i) {
        const unsigned char c = (unsigned char)cstr[i];
        if (!url_unreserved_char(c)) {
            if (!dest) {
                dest = rb_str_buf_new(len);
            }

            rb_str_cat(dest, cstr + beg, i - beg);
            beg = i + 1;

            if (plus_escape && c == ' ') {
                rb_str_cat_cstr(dest, "+");
            }
            else {
                buf[1] = upper_hexdigits[(c >> 4) & 0xf];
                buf[2] = upper_hexdigits[c & 0xf];
                rb_str_cat(dest, buf, 3);
            }
        }
    }

    if (dest) {
        rb_str_cat(dest, cstr + beg, len - beg);
        preserve_original_state(str, dest);
        return dest;
    }
    else {
        return rb_str_dup(str);
    }
}

static VALUE
optimized_unescape(VALUE str, VALUE encoding, int unescape_plus)
{
    rb_len_t i, len, beg = 0;
    VALUE dest = 0;
    const char *cstr;
    rb_encoding *enc = rb_to_encoding(encoding);
    int cr, origenc, encidx = rb_enc_to_index(enc);

    len  = RSTRING_LEN(str);
    cstr = RSTRING_PTR(str);

    for (i = 0; i < len; ++i) {
        char buf[1];
        const char c = cstr[i];
        int clen = 0;
        if (c == '%') {
            if (i + 3 > len) break;
            if (!ISXDIGIT(cstr[i+1])) continue;
            if (!ISXDIGIT(cstr[i+2])) continue;
            buf[0] = ((char_to_number(cstr[i+1]) << 4)
                      | char_to_number(cstr[i+2]));
            clen = 2;
        }
        else if (unescape_plus && c == '+') {
            buf[0] = ' ';
        }
        else {
            continue;
        }

        if (!dest) {
            dest = rb_str_buf_new(len);
        }

        rb_str_cat(dest, cstr + beg, i - beg);
        i += clen;
        beg = i + 1;

        rb_str_cat(dest, buf, 1);
    }

    if (dest) {
        rb_str_cat(dest, cstr + beg, len - beg);
        preserve_original_state(str, dest);
        cr = ENC_CODERANGE_UNKNOWN;
    }
    else {
        dest = rb_str_dup(str);
        cr = ENC_CODERANGE(str);
    }
    origenc = rb_enc_get_index(str);
    if (origenc != encidx) {
        rb_enc_associate_index(dest, encidx);
        if (!ENC_CODERANGE_CLEAN_P(rb_enc_str_coderange(dest))) {
            rb_enc_associate_index(dest, origenc);
            if (cr != ENC_CODERANGE_UNKNOWN)
                ENC_CODERANGE_SET(dest, cr);
        }
    }
    return dest;
}

/*
 *  call-seq:
 *     CGI.escapeHTML(string) -> string
 *
 *  Returns HTML-escaped string.
 *
 */
static VALUE
cgiesc_escape_html(VALUE self, VALUE str)
{
    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        return optimized_escape_html(str);
    }
    else {
        return rb_call_super(1, &str);
    }
}

/*
 *  call-seq:
 *     CGI.unescapeHTML(string) -> string
 *
 *  Returns HTML-unescaped string.
 *
 */
static VALUE
cgiesc_unescape_html(VALUE self, VALUE str)
{
    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        return optimized_unescape_html(str);
    }
    else {
        return rb_call_super(1, &str);
    }
}

/*
 *  call-seq:
 *     CGI.escape(string) -> string
 *
 *  Returns URL-escaped string (+application/x-www-form-urlencoded+).
 *
 */
static VALUE
cgiesc_escape(VALUE self, VALUE str)
{
    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        return optimized_escape(str, 1);
    }
    else {
        return rb_call_super(1, &str);
    }
}

static VALUE
accept_charset(int argc, VALUE *argv, VALUE self)
{
    if (argc > 0)
        return argv[0];
    return rb_cvar_get(CLASS_OF(self), id_accept_charset);
}

/*
 *  call-seq:
 *     CGI.unescape(string, encoding=@@accept_charset) -> string
 *
 *  Returns URL-unescaped string (+application/x-www-form-urlencoded+).
 *
 */
static VALUE
cgiesc_unescape(int argc, VALUE *argv, VALUE self)
{
    VALUE str = (rb_check_arity(argc, 1, 2), argv[0]);

    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        VALUE enc = accept_charset(argc-1, argv+1, self);
        return optimized_unescape(str, enc, 1);
    }
    else {
        return rb_call_super(argc, argv);
    }
}

/*
 *  call-seq:
 *     CGI.escapeURIComponent(string) -> string
 *
 *  Returns URL-escaped string following RFC 3986.
 *
 */
static VALUE
cgiesc_escape_uri_component(VALUE self, VALUE str)
{
    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        return optimized_escape(str, 0);
    }
    else {
        return rb_call_super(1, &str);
    }
}

/*
 *  call-seq:
 *     CGI.unescapeURIComponent(string, encoding=@@accept_charset) -> string
 *
 *  Returns URL-unescaped string following RFC 3986.
 *
 */
static VALUE
cgiesc_unescape_uri_component(int argc, VALUE *argv, VALUE self)
{
    VALUE str = (rb_check_arity(argc, 1, 2), argv[0]);

    StringValue(str);

    if (rb_enc_str_asciicompat_p(str)) {
        VALUE enc = accept_charset(argc-1, argv+1, self);
        return optimized_unescape(str, enc, 0);
    }
    else {
        return rb_call_super(argc, argv);
    }
}

void
Init_escape(void)
{
#ifdef HAVE_RB_EXT_RACTOR_SAFE
    rb_ext_ractor_safe(true);
#endif

    id_accept_charset = rb_intern_const("@@accept_charset");
    InitVM(escape);
}

void
InitVM_escape(void)
{
    rb_cCGI       = rb_define_class("CGI", rb_cObject);
    rb_mEscapeExt = rb_define_module_under(rb_cCGI, "EscapeExt");
    rb_mEscape    = rb_define_module_under(rb_cCGI, "Escape");
    rb_define_method(rb_mEscapeExt, "escapeHTML", cgiesc_escape_html, 1);
    rb_define_method(rb_mEscapeExt, "unescapeHTML", cgiesc_unescape_html, 1);
    rb_define_method(rb_mEscapeExt, "escapeURIComponent", cgiesc_escape_uri_component, 1);
    rb_define_alias(rb_mEscapeExt, "escape_uri_component", "escapeURIComponent");
    rb_define_method(rb_mEscapeExt, "unescapeURIComponent", cgiesc_unescape_uri_component, -1);
    rb_define_alias(rb_mEscapeExt, "unescape_uri_component", "unescapeURIComponent");
    rb_define_method(rb_mEscapeExt, "escape", cgiesc_escape, 1);
    rb_define_method(rb_mEscapeExt, "unescape", cgiesc_unescape, -1);
    rb_prepend_module(rb_mEscape, rb_mEscapeExt);
    rb_extend_object(rb_cCGI, rb_mEscapeExt);
}
