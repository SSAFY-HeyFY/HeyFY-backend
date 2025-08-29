package com.ssafy.ssashinsa.heyfy.fcm.scheduler;

import com.ssafy.ssashinsa.heyfy.fcm.service.SchedulerService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationScheduler {
    private final SchedulerService schedulerService;

    @Scheduled(cron = "${notification.schedule.cron}", zone = "Asia/Seoul")
    public void sendDailyExchangeRateNotification() {
        log.info("스케줄러 실행: 환율 정보를 가져와 모든 사용자에게 알림을 보냅니다.");
        schedulerService.sendExchangeRateInfo();
    }
}