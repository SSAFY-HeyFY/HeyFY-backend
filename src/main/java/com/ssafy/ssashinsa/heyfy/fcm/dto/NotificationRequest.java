package com.ssafy.ssashinsa.heyfy.fcm.dto;

import lombok.*;

@Getter
@ToString
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class NotificationRequest {
    private String token;
    private String title;
    private String body;

    @Builder(toBuilder = true)
    public NotificationRequest(String token, String title, String body) {
        this.token = token;
        this.title = title;
        this.body = body;
    }
}
