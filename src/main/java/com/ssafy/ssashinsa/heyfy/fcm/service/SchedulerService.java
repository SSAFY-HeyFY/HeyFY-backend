package com.ssafy.ssashinsa.heyfy.fcm.service;

import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRateStatusResponseDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class SchedulerService {

    private final NotificationService fcmService;
    private final FastApiClient fastApiClient;

    public void sendExchangeRateInfo() {
        FastApiRateStatusResponseDto response = fastApiClient.getRateStatus();
        if (response != null && response.getMessage() != null && !response.getMessage().isEmpty()) {
            String title = "Today's Exchange Rate Information";
            String body = response.getMessage();
            fcmService.sendNotificationToAll(title, body);
        }
    }
}