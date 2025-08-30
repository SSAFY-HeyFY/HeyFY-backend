package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.exchange.docs.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.*;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeRateService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@Tag(name = "환율 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange-rate")
public class ExchangeRateApiController {
    private final ExchangeRateService exchangeRateService;

    @ExchangeRatePageDocs
    @GetMapping("/page")
    public ResponseEntity<ExchangeRatePageResponseDto> getExchangeRatePage() {
        log.info("환율 페이지 조회 요청");
        ExchangeRatePageResponseDto response = exchangeRateService.getExchangeRatePage();
        return ResponseEntity.ok(response);
    }

    @ExchangeRateHistoriesDocs
    @GetMapping("/histories")
    public ResponseEntity<ExchangeRateHistoriesResponseDto> getExchangeRateHistories() {
        log.info("환전 그래프 데이터 조회 요청");
        return ResponseEntity.ok(exchangeRateService.getExchangeRateHistories());
    }

    @ExchangeRateCurrentDocs
    @GetMapping("/current")
    public ResponseEntity<RealTimeRateGroupResponseDto> getCurrentExchangeRates() {
        log.info("실시간 환율 조회 요청");
        return ResponseEntity.ok(exchangeRateService.getRealTimeRate());
    }

    @ExchangeRatePredictionDocs
    @GetMapping("/prediction")
    public ResponseEntity<PredictionResponseDto> getPrediction() {
        log.info("환율 예측 조회 요청");
        return ResponseEntity.ok(exchangeRateService.getPredictionSummary());
    }

    @ExchangeRateTuitionDocs
    @GetMapping("/tuition")
    public ResponseEntity<TuitionResponseDto> getTuition() {
        log.info("환전 일자 추천 요청");
        return ResponseEntity.ok(exchangeRateService.getTuition());
    }
}
