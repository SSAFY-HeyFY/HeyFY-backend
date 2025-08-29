package com.ssafy.ssashinsa.heyfy.fcm.service;

import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRateStatusResponseDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class SchedulerService {

    private final NotificationService notificationService;
    private final FastApiClient fastApiClient;
    public void sendExchangeRateInfo() {
        log.info("환율 정보를 가져와 모든 사용자에게 알림을 보냅니다.");
        FastApiRateStatusResponseDto response = fastApiClient.getRateStatus();
        if (response != null && response.getMessage() != null && !response.getMessage().isEmpty()) {
            String title = "Today's Exchange Rate Information";
            String body = response.getMessage();
            notificationService.sendNotificationToAll(title, body);
        } else {
            log.warn("환율 정보를 가져오지 못했습니다.");
        }
    }
}