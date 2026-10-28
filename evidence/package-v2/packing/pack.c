/* SPDX-License-Identifier: MIT
 * Fixed-stride canonical packing. No CUDA, hashing, or state normalization.
 * Accepts arbitrary native-endian uint32 values; emits their low three bytes.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void put_le32(unsigned char *p, uint32_t x) {
    p[0]=(unsigned char)x; p[1]=(unsigned char)(x>>8);
    p[2]=(unsigned char)(x>>16); p[3]=(unsigned char)(x>>24);
}

int bram_pack_u32_le24(const void *input, size_t words, void *output,
                      size_t output_bytes) {
    if (words > SIZE_MAX/4 || words > SIZE_MAX/3) return 1;
    const size_t inbytes=words*4, outbytes=words*3;
    if (output_bytes != outbytes) return 2;
    if (words == 0) return 0;
    if (!input || !output) return 3;
    uintptr_t a=(uintptr_t)input, b=(uintptr_t)output;
    if (a > UINTPTR_MAX-inbytes || b > UINTPTR_MAX-outbytes) return 4;
    if (a < b+outbytes && b < a+inbytes) return 5;
    const unsigned char *src=(const unsigned char *)input;
    unsigned char *dst=(unsigned char *)output;
    size_t i=0;
    for (; words-i >= 4; i+=4) {
        uint32_t x[4];
        memcpy(x, src+4*i, sizeof x);
        /* Masks preserve low24 identity even for noncanonical uint32 inputs. */
        put_le32(dst+3*i, (x[0]&UINT32_C(0xffffff)) | (x[1]<<24));
        put_le32(dst+3*i+4, ((x[1]>>8)&UINT32_C(0xffff)) | (x[2]<<16));
        put_le32(dst+3*i+8, ((x[2]>>16)&UINT32_C(0xff)) | (x[3]<<8));
    }
    for (; i<words; ++i) {
        uint32_t x; memcpy(&x, src+4*i, sizeof x);
        dst[3*i]=(unsigned char)x; dst[3*i+1]=(unsigned char)(x>>8);
        dst[3*i+2]=(unsigned char)(x>>16);
    }
    return 0;
}
