package com.ssafy.ssashinsa.heyfy.exchange.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import static org.junit.jupiter.api.Assertions.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.EntireExchangeRateResponseDto;

@SpringBootTest
class ExchangeRateApiServiceTest {

    @Autowired
    private ExchangeRateApiService exchangeRateApiService;

    @Test
    void 전체_환율_조회() {
        // 실제 API 호출
        EntireExchangeRateResponseDto response = exchangeRateApiService.getExchangeRate();
        assertNotNull(response, "API response should not be null");
        String responseString = response.toString();
        System.out.println("API Response: " + responseString);
        assertTrue(responseString.contains("exchangeRate") || responseString.contains("rate"), "Response should contain 'exchangeRate' or 'rate'");
    }
}