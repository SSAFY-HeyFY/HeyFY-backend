package com.ssafy.ssashinsa.heyfy.fastapi;

import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRealTimeRatesDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class FastApiClientTest {

    @Autowired
    FastApiClient fastApiClient;

    @Test
    @DisplayName("FastApiClient 실시간 환율 API 실행 테스트")
    void getRealTimeRatesTest() {
        try {
            FastApiRealTimeRatesDto result = fastApiClient.getRealTimeRates();
            System.out.println("getRealTimeRates result: " + result);
        } catch (Exception e) {
            System.out.println("getRealTimeRates error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("FastApiClient 환율 그래프 API 실행 테스트")
    void getRateGraphTest() {
        try {
            var result = fastApiClient.getRateGraph();
            System.out.println("getRateGraph result: " + result);
        } catch (Exception e) {
            System.out.println("getRateGraph error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("FastApiClient 환율 분석 API 실행 테스트")
    void getRateAnalysisTest() {
        try {
            var result = fastApiClient.getRateAnalysis();
            System.out.println("getRateAnalysis result: " + result);
        } catch (Exception e) {
            System.out.println("getRateAnalysis error: " + e.getMessage());
        }
    }
}
