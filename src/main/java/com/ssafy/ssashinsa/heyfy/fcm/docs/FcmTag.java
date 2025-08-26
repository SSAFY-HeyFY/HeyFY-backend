package com.ssafy.ssashinsa.heyfy.fcm.docs;

import io.swagger.v3.oas.annotations.tags.Tag;

import java.lang.annotation.*;

@Target(ElementType.TYPE)   // 클래스에 붙일 수 있게
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Tag(
        name = "FCM Tokens",
        description = "사용자 FCM 토큰 등록 및 삭제 API"
)
public @interface FcmTag {}
