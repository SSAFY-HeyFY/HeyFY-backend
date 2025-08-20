package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.exchange.dto.*;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeRateService;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@ErrorsCommonDocs
@Tag(name = "모집글 참여 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange-rate")
public class ExchangeRateApiController {
    private final ExchangeRateService exchangeRateService;

    @GetMapping("/page")
    public ResponseEntity<ExchangeRatePageResponseDto> getExchangeRatePage() {
        ExchangeRatePageResponseDto response = exchangeRateService.getExchangeRatePage();
        return ResponseEntity.ok(response);
    }

    @GetMapping("/histories")
    public ResponseEntity<ExchangeRateHistoriesDto> getExchangeRateHistories(@RequestParam(defaultValue="USD")String currency, @RequestParam(defaultValue = "29") Integer day) {
        // 기본값: USD, 29일
        return ResponseEntity.ok(exchangeRateService.getExchangeRateHistories(currency, day));
    }

    @GetMapping("/current")
    public ResponseEntity<ExchangeRateGroupDto> getCurrentExchangeRates() {
        return ResponseEntity.ok(exchangeRateService.getCurrentExchangeRates());
    }

    @GetMapping("/prediction")
    public ResponseEntity<PredictionDto> getPrediction() {
        return ResponseEntity.ok(exchangeRateService.getPrediction());
    }

    @GetMapping("/tuition")
    public ResponseEntity<TuitionDto> getTuition() {
        return ResponseEntity.ok(exchangeRateService.getTuition());
    }
}
