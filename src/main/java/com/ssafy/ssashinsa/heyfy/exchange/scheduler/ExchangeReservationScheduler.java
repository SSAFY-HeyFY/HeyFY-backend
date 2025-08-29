package com.ssafy.ssashinsa.heyfy.exchange.scheduler;

import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeReservationRepository;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeReservationService;
import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExchangeReservationScheduler {

    private final FastApiClient fastApiClient;
    private final ExchangeReservationRepository exchangeReservationRepository;
    private final ExchangeReservationService exchangeReservationService;
    private final FcmService fcmService;

    @Scheduled(cron = "${reservation.schedule.cron}", zone = "Asia/Seoul")
    public void checkExchangeRate(){
        log.info("스케줄러 실행: 환전 예약 확인");

        try{
            List<FcmToken> fcmTokens = exchangeReservationService.exchangeToForeignBatch();
            fcmService.sendNotificationByFcmTokenList("Exchange Completed", "Your reserved currency exchange has been completed.", fcmTokens);

        }catch (Exception e) {
            log.error("외부 API 호출 또는 알림 발송 중 오류가 발생했습니다.", e);

        }
    }

}
