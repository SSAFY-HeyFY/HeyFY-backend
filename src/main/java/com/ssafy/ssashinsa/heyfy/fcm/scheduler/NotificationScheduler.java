package com.ssafy.ssashinsa.heyfy.fcm.scheduler;

import com.google.firebase.messaging.FirebaseMessagingException;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationScheduler {

    private final FcmTokenRepository fcmTokenRepository;
    private final FcmService fcmService;

    @Scheduled(fixedRate = 300000)
    public void sendNotificationToAllLoggedInUsers() {
        log.info("스케줄러 실행: 모든 로그인된 사용자에게 알림을 보냅니다.");

        List<FcmToken> allTokens = fcmTokenRepository.findAll();

        if (allTokens.isEmpty()) {
            log.info("알림을 보낼 대상이 없습니다.");
            return;
        }

        String title = "HeyFY 서비스 알림";
        String body = "로그인된 사용자에게만 보내는 테스트 알림입니다! 👋";


        for (FcmToken fcmToken : allTokens) {
            try {
                fcmService.sendNotification(fcmToken.getToken(), title, body);
            } catch (FirebaseMessagingException e) {
                log.error("알림 발송 실패. Token: {}, Error: {}", fcmToken.getToken(), e.getMessage());
            }
        }
        log.info("{}명에게 알림 발송을 시도했습니다.", allTokens.size());
    }
}