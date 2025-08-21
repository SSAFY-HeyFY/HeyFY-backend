package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeService;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
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
}
