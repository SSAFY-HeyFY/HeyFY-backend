package com.ssafy.ssashinsa.heyfy.fcm.docs;

import io.swagger.v3.oas.annotations.tags.Tag;

import java.lang.annotation.*;

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Tag(
        name = "FCM Notification",
        description = "알림 보내는 API (인증 필요 없음)"
)
public @interface NotificationTag {}
