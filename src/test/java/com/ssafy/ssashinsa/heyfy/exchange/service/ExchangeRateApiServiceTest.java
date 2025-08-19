package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeRateService;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ExchangeRateGroupDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanCommonResponseHeaderDto;
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
        EntireExchangeRateResponseDto response = exchangeRateService.getExchangeRate();
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
}