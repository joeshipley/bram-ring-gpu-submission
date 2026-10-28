/* SPDX-License-Identifier: MIT: CPU memory-boundary/undefined-behavior checks. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
int bram_pack_u32_le24(const void *,size_t,void *,size_t);
static void require(int ok) { if(!ok) abort(); }
int main(void) {
    const size_t counts[]={0,1,2,3,4,5,7,31,32,33,767,768,769,768*5,768*1840};
    for(size_t t=0;t<sizeof counts/sizeof counts[0];++t) {
        size_t n=counts[t];
        for(size_t offset=0;offset<8;++offset) {
            unsigned char *src=malloc(4*n+offset ? 4*n+offset : 1);
            unsigned char *dst=malloc(3*n+offset ? 3*n+offset : 1);
            require(src && dst);
            memset(src,0xa5,4*n+offset);memset(dst,0x5a,3*n+offset);
            for(size_t i=0;i<n;++i) {uint32_t x=(uint32_t)(i*UINT32_C(2654435761));memcpy(src+offset+4*i,&x,4);}
            require(bram_pack_u32_le24(src+offset,n,dst+offset,3*n)==0);
            for(size_t i=0;i<n;++i) {uint32_t x;memcpy(&x,src+offset+4*i,4);
                for(size_t j=0;j<3;++j)require(dst[offset+3*i+j]==(unsigned char)(x>>(8*j)));}
            for(size_t i=0;i<offset;++i)require(src[i]==0xa5 && dst[i]==0x5a);
            if(n)require(bram_pack_u32_le24(src+offset,n,src+offset,3*n)==5);
            free(src);free(dst);
        }
    }
    unsigned char x[16]={0};
    require(bram_pack_u32_le24(NULL,1,x,3)==3);
    require(bram_pack_u32_le24(x,1,NULL,3)==3);
    require(bram_pack_u32_le24(x,1,x+8,2)==2);
    require(bram_pack_u32_le24(x,SIZE_MAX,x,0)==1);
    require(bram_pack_u32_le24(NULL,0,NULL,0)==0);
    puts("CPU address/undefined sanitizer and exact-size tail checks passed");
    return 0;
}
