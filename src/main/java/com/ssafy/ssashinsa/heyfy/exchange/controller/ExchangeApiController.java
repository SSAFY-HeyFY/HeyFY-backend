package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AIPredictionResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.ExchangePageResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.HistoricalAnalysisResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeService;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@ErrorsCommonDocs
@Tag(name = "환전 페이지 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange")
public class ExchangeApiController {
    private final ExchangeService exchangeService;

    @GetMapping("/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getAccountBalance(@AuthUser UserDetails userDetails) {
        return ResponseEntity.ok(
                exchangeService.getAccountBalance(userDetails.getUsername())
        );
    }

    @GetMapping("/foreign/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getForeignAccountBalance(@AuthUser UserDetails userDetails) {
        return ResponseEntity.ok(
                exchangeService.getForeignAccountBalance(userDetails.getUsername())
        );
    }

    @GetMapping("/ai-prediction")
    public ResponseEntity<AIPredictionResponseDto> getAIPrediction(){
        return ResponseEntity.ok(
                exchangeService.getExchangeRateAIPrediction()
        );
    }

    @GetMapping("/historical-analysis")
    public ResponseEntity<HistoricalAnalysisResponseDto> getHistoricalAnalysis(){
        return ResponseEntity.ok(
                exchangeService.getHistoricalAnalysis()
        );
    }

    @GetMapping("/page")
    public ResponseEntity<ExchangePageResponseDto> getExchangePage(@AuthUser UserDetails userDetails){
        return ResponseEntity.ok(
                exchangeService.getExchangePage(userDetails.getUsername())
        );
    }


}
