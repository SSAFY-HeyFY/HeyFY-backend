package com.ssafy.ssashinsa.heyfy.fcm.scheduler;

import com.google.firebase.messaging.FirebaseMessagingException;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.fcm.dto.ExchangeRateResponse;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationScheduler {
    private final FcmService fcmService;
    private final RestTemplate restTemplate;

    @Scheduled(cron = "${notification.schedule.cron}", zone = "Asia/Seoul")
    public void sendDailyExchangeRateNotification() {
        log.info("스케줄러 실행: 환율 정보를 가져와 모든 사용자에게 알림을 보냅니다.");

        try {
            String apiUrl = "http://114.199.133.118:8888/api/push/rate-status";
            ExchangeRateResponse response = restTemplate.getForObject(apiUrl, ExchangeRateResponse.class);

            if (response != null && response.getMessage() != null && !response.getMessage().isEmpty()) {
                String title = "Today's Exchange Rates";
                String body = response.getMessage();


                fcmService.sendNotificationToAll(title, body);
            } else {
                log.warn("API로부터 유효한 메시지를 받지 못했습니다.");
            }

        } catch (Exception e) {
            // API 호출 실패 등 예외 발생 시 로그 기록
            log.error("외부 API 호출 또는 알림 발송 중 오류가 발생했습니다.", e);
        }
    }
}