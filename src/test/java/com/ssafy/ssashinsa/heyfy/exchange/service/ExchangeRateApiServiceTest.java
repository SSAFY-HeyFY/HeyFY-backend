package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeRateRepository;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
@SpringBootTest
@ActiveProfiles("local")
class ExchangeRateApiServiceTest {

    @Autowired
    private ExchangeRateService exchangeRateService;
    @Autowired
    private ShinhanApiClient shinhanApiClient;
    @Autowired
    private ShinhanApiUtil shinhanApiUtil;
    @Autowired
    private ExchangeRateRepository exchangeRateRepository;

    @Test
    void 전체_환율_조회() {
        // 실제 API 호출
        EntireExchangeRateResponseDto response = exchangeRateService.getExchangeRateFromExternalApi();
        assertNotNull(response, "API response should not be null");
        String responseString = response.toString();
        System.out.println("API Response: " + responseString);
        assertTrue(responseString.contains("exchangeRate") || responseString.contains("rate"), "Response should contain 'exchangeRate' or 'rate'");
    }

    @Test
    public void 한달간_환율_조회() {
        // given
        String currencyCode = "USD";
        LocalDate endDate = LocalDate.now().minusDays(1);
        LocalDate startDate = endDate.minusDays(29);

        List<ExchangeRate> mockRates =
                startDate.datesUntil(endDate.plusDays(1))
                        .map(date -> ExchangeRate.builder()
                                .id(1L)
                                .currencyCode(currencyCode)
                                .country("United States")
                                .rate(BigDecimal.valueOf(1300 + date.getDayOfMonth())) // 샘플 값
                                .unit(1)
                                .baseDate(date)
                                .build()
                        ).toList();
        when(exchangeRateRepository.findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
                currencyCode, startDate, endDate
        )).thenReturn(mockRates);

        // when
        List<ExchangeRate> result = exchangeRateService.getLast30DaysRates(currencyCode);

        // then
        assertThat(result).hasSize(30);
        assertThat(result.get(0).getBaseDate()).isEqualTo(startDate);
        assertThat(result.get(29).getBaseDate()).isEqualTo(endDate);

        verify(exchangeRateRepository, times(1))
                .findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(currencyCode, startDate, endDate);
    }

    @Test
    void 환율_페이지_통합_조회() throws Exception {
        // when
        var result = exchangeRateService.getExchangeRatePage();
        // then
        assertNotNull(result, "ExchangeRatePageResponseDto should not be null");
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        String json = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        System.out.println("\n===== ExchangeRatePageResponseDto (JSON) =====\n" + json + "\n==============================================\n");
        // 추가적으로 각 필드별 null 체크 등도 가능
        assertNotNull(result.getExchangeRateHistories(), "exchangeRateHistories should not be null");
        assertNotNull(result.getLatestExchangeRate(), "latestExchangeRate should not be null");
        assertNotNull(result.getPrediction(), "prediction should not be null");
        assertNotNull(result.getTuition(), "tuition should not be null");
    }
}