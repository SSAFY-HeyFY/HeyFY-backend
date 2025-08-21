package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.exchange.docs.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseRecDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeService;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@Slf4j
@ErrorsCommonDocs
@Tag(name = "환전 페이지 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange")
public class ExchangeApiController {
    private final ExchangeService exchangeService;

    @AccountBalanceDocs
    @GetMapping("/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getAccountBalance(@AuthUser UserDetails userDetails) {
        return ResponseEntity.ok(
                exchangeService.getAccountBalance(userDetails.getUsername())
        );
    }

    @ForeignAccountBalanceDocs
    @GetMapping("/foreign/account-balance")
    public ResponseEntity<AccountBalanceResponseDto> getForeignAccountBalance(@AuthUser UserDetails userDetails) {
        return ResponseEntity.ok(
                exchangeService.getForeignAccountBalance(userDetails.getUsername())
        );
    }

    @AIPredictionDocs
    @GetMapping("/ai-prediction")
    public ResponseEntity<AIPredictionResponseDto> getAIPrediction(){
        return ResponseEntity.ok(
                exchangeService.getExchangeRateAIPrediction()
        );
    }


    @HistoricalAnalysisDocs
    @GetMapping("/historical-analysis")
    public ResponseEntity<HistoricalAnalysisResponseDto> getHistoricalAnalysis(){
        return ResponseEntity.ok(
                exchangeService.getHistoricalAnalysis()
        );
    }


    @ExchangePageDocs
    @GetMapping("/page")
    public ResponseEntity<ExchangePageResponseDto> getExchangePage(@AuthUser UserDetails userDetails){
        return ResponseEntity.ok(
                exchangeService.getExchangePage(userDetails.getUsername())
        );
    }

    @ExchangeDocs
    @PostMapping
    public ResponseEntity<ShinhanExchangeResponseRecDto> exchangeToForeign(@AuthUser UserDetails userDetails, @RequestBody ExchangeRequestDto exchangeRequestDto) {
        return ResponseEntity.ok(exchangeService.exchangeToForeign(userDetails.getUsername(), exchangeRequestDto));
    }

    @ExchangeForeignDocs
    @PostMapping("/foreign")
    public ResponseEntity<ShinhanExchangeResponseRecDto> exchangeFromForeign(@AuthUser UserDetails userDetails, @RequestBody ExchangeRequestDto exchangeRequestDto) {
        return ResponseEntity.ok(exchangeService.exchangeFromForeign(userDetails.getUsername(), exchangeRequestDto));
    }
}
