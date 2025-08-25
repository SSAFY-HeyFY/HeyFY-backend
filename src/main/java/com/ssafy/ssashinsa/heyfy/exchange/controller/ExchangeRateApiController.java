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
@Tag(name = "환율 페이지 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange-rate")
public class ExchangeRateApiController {
    private final ExchangeRateService exchangeRateService;

    @ExchangeRatePageDocs
    @GetMapping("/page")
    public ResponseEntity<ExchangeRatePageResponseDto> getExchangeRatePage() {
        ExchangeRatePageResponseDto response = exchangeRateService.getExchangeRatePage();
        return ResponseEntity.ok(response);
    }

    @ExchangeRateHistoriesDocs
    @GetMapping("/histories")
    public ResponseEntity<ExchangeRateHistoriesResponseDto> getExchangeRateHistories() {
        // 기본값: USD, 29일
        return ResponseEntity.ok(exchangeRateService.getExchangeRateHistories());
    }

    @ExchangeRateCurrentDocs
    @GetMapping("/current")
    public ResponseEntity<RealTimeRateGroupResponseDto> getCurrentExchangeRates() {
        return ResponseEntity.ok(exchangeRateService.getRealTimeRate());
    }

    @ExchangeRatePredictionDocs
    @GetMapping("/prediction")
    public ResponseEntity<PredictionResponseDto> getPrediction() {
        return ResponseEntity.ok(exchangeRateService.getPrediction());
    }

    @ExchangeRateTuitionDocs
    @GetMapping("/tuition")
    public ResponseEntity<TuitionResponseDto> getTuition() {
        return ResponseEntity.ok(exchangeRateService.getTuition());
    }
}
