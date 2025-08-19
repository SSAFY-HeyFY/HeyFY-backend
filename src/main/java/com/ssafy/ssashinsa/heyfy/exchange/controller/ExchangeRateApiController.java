package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;

import com.ssafy.ssashinsa.heyfy.exchange.dto.ExchangeRatePageResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeRateService;

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
}
