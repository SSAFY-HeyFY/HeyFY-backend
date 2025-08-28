package com.ssafy.ssashinsa.heyfy.fcm.scheduler;

import com.ssafy.ssashinsa.heyfy.fcm.dto.ExchangeRateResponse;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationScheduler {

    private final FcmTokenRepository fcmTokenRepository;
    private final FcmService fcmService;
    private final RestTemplate restTemplate;

    @Scheduled(cron = "0 0 7 * * *", zone = "Asia/Seoul")
    public void sendDailyExchangeRateNotification() {
        log.info("스케줄러 실행: 환율 정보를 가져와 모든 사용자에게 알림을 보냅니다.");

        try {
            String apiUrl = "http://114.199.133.118:8888/api/push/rate-status";
            ExchangeRateResponse response = restTemplate.getForObject(apiUrl, ExchangeRateResponse.class);

            log.info("API 응답: {}", response);
            if (response != null && response.getMessage() != null && !response.getMessage().isEmpty()) {
                String title = "Today's Exchange Rate Information";
                String body = response.getMessage();

                fcmService.sendNotificationToAll(title, body);
            } else {
                log.warn("API로부터 유효한 메시지를 받지 못했습니다.");
            }

        } catch (Exception e) {
            log.error("외부 API 호출 또는 알림 발송 중 오류가 발생했습니다.", e);
        }
    }
}