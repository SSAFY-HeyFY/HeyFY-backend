package com.ssafy.ssashinsa.heyfy.fcm.service;

import com.google.firebase.messaging.*;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Slf4j
@RequiredArgsConstructor
public class NotificationService {

    private final UserFcmService userFcmService;
    private final FcmTokenRepository fcmTokenRepository;

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
            userFcmService.handleSendFailure(token, e);
            throw e;
        }
    }

    public void sendNotificationToAll(String title, String body) {
        List<FcmToken> allTokens = fcmTokenRepository.findAll(); // 모든 토큰 조회

        if (allTokens.isEmpty()) {
            log.info("알림을 보낼 대상이 없습니다.");
            return;
        }

        for (FcmToken fcmToken : allTokens) {
            try {
                this.sendNotification(fcmToken.getToken(), title, body);
            } catch (FirebaseMessagingException e) {
                log.error("알림 발송 실패. Token: {}, Error: {}", fcmToken.getToken(), e.getMessage());
            }
        }
        log.info("{}명에게 전체 알림 발송을 시도했습니다.", allTokens.size());
    }
}
