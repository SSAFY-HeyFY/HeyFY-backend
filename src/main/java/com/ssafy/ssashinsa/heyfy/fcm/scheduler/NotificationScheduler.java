package com.ssafy.ssashinsa.heyfy.fcm.scheduler;

import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRateStatusResponseDto;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationScheduler {

    private final FcmTokenRepository fcmTokenRepository;
    private final FcmService fcmService;
    private final FastApiClient fastApiClient;

    @Scheduled(cron = "0 0 7 * * *", zone = "Asia/Seoul")
    public void sendDailyExchangeRateNotification() {
        log.info("스케줄러 실행: 환율 정보를 가져와 모든 사용자에게 알림을 보냅니다.");
        FastApiRateStatusResponseDto response = fastApiClient.getRateStatus();
        if (response != null && response.getMessage() != null && !response.getMessage().isEmpty()) {
            String title = "Today's Exchange Rate Information";
            String body = response.getMessage();

            fcmService.sendNotificationToAll(title, body);
        }

    }
}