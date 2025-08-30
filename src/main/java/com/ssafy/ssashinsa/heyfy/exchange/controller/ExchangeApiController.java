package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.docs.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.*;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@Slf4j
@Tag(name = "환전 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange")
public class ExchangeApiController {
    private final ExchangeService exchangeService;

    @AccountBalanceDocs
    @GetMapping("/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getAccountBalance(@AuthUser UserDetails userDetails) {
        log.info("계좌 잔액 조회 요청: {}", userDetails.getUsername());
        return ResponseEntity.ok(
                exchangeService.getAccountBalance(userDetails.getUsername())
        );
    }

    @ForeignAccountBalanceDocs
    @GetMapping("/foreign/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getForeignAccountBalance(@AuthUser UserDetails userDetails) {
        log.info("외화 계좌 잔액 조회 요청: {}", userDetails.getUsername());
        return ResponseEntity.ok(
                exchangeService.getForeignAccountBalance(userDetails.getUsername())
        );
    }

    @AIPredictionDocs
    @GetMapping("/ai-prediction")
    public ResponseEntity<AIPredictionResponseDto> getAIPrediction() {
        log.info("AI 환율 예측 조회 요청");
        return ResponseEntity.ok(
                exchangeService.getExchangeRateAIPrediction()
        );
    }


    @HistoricalAnalysisDocs
    @GetMapping("/historical-analysis")
    public ResponseEntity<HistoricalAnalysisResponseDto> getHistoricalAnalysis() {
        log.info("환율 기록 분석 조회 요청");
        return ResponseEntity.ok(
                exchangeService.getHistoricalAnalysis()
        );
    }

    @RateAnalysisDocs
    @GetMapping("/analysis")
    public ResponseEntity<RateAnalysisResponseDto> getRateAnalysis() {
        log.info("환율 분석 조회 요청");
        return ResponseEntity.ok(
                exchangeService.getRateAnalysis()
        );
    }

    @ExchangePageDocs
    @GetMapping("/page")
    public ResponseEntity<ExchangePageResponseDto> getExchangePage(@AuthUser UserDetails userDetails) {
        log.info("환전 화면 조회 요청 : {}", userDetails.getUsername());
        return ResponseEntity.ok(
                exchangeService.getExchangePage(userDetails.getUsername())
        );
    }

    @ExchangeDocs
    @PostMapping
    public ResponseEntity<ExchangeResponseDto> exchangeToForeign(@AuthUser UserDetails userDetails, @RequestBody ExchangeRequestDto exchangeRequestDto) {
        log.info("원화 -> 외화 환전 요청 : {}", userDetails.getUsername());
        try{
            ExchangeResponseDto response = exchangeService.exchangeToForeign(userDetails.getUsername(), exchangeRequestDto);
            return ResponseEntity.ok(response);
        } catch (CustomException e){
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                ExchangeResponseDto response = ExchangeResponseDto.builder().
                        depositAccountBalance(null).
                        withdrawalAccountBalance(null).
                        transactionBalance(null).
                        isCorrect(false).
                        build();
                return ResponseEntity.ok(response);
            }
            throw e;
        }

    }

    @ExchangeForeignDocs
    @PostMapping("/foreign")
    public ResponseEntity<ExchangeResponseDto> exchangeFromForeign(@AuthUser UserDetails userDetails, @RequestBody ExchangeRequestDto exchangeRequestDto) {
        log.info("외화 -> 원화 환전 요청 : {}", userDetails.getUsername());
        try{
            ExchangeResponseDto response = exchangeService.exchangeFromForeign(userDetails.getUsername(), exchangeRequestDto);
            return ResponseEntity.ok(response);
        } catch (CustomException e){
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                ExchangeResponseDto response = ExchangeResponseDto.builder().
                        depositAccountBalance(null).
                        withdrawalAccountBalance(null).
                        transactionBalance(null).
                        isCorrect(false).
                        build();
                return ResponseEntity.ok(response);
            }
            throw e;
        }
    }
}
