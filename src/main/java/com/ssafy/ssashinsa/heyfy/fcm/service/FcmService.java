// src/main/java/com/ssafy/ssashinsa/heyfy/fcm/service/FcmService.java
package com.ssafy.ssashinsa.heyfy.fcm.service;

import com.google.firebase.messaging.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class FcmService {

    private final UserFcmService userFcmService; // 추가 주입

    public void sendNotification(String token, String title, String body) throws FirebaseMessagingException {
        Notification notification = Notification.builder()
                .setTitle(title)
                .setBody(body)
                .build();

        Message message = Message.builder()
                .setToken(token)
                .setNotification(notification)
                .build();

        try {
            FirebaseMessaging.getInstance().send(message);
        } catch (FirebaseMessagingException e) {
            userFcmService.handleSendFailure(token, e); // 실패 시 DB 정리
            throw e;
        }
    }
}
