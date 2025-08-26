package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.ExchangeRateHistoriesResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.RealTimeRateGroupResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeRateRepository;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.*;


@SpringBootTest
@ExtendWith(MockitoExtension.class)
@ActiveProfiles("local")
class ExchangeRateApiServiceTest {

    @InjectMocks
    private ExchangeRateService exchangeRateService;
    @Mock
    private ExchangeRateRepository exchangeRateRepository;
    @Mock
    private ShinhanApiClient shinhanApiClient;
    @Mock
    private ShinhanApiUtil shinhanApiUtil;

    @Test
    void testGetAllExchangeRates() {
        // given: WebClient mock 체이닝 설정
        var postSpec = mock(org.springframework.web.reactive.function.client.WebClient.RequestBodyUriSpec.class);
        var headersSpec = mock(org.springframework.web.reactive.function.client.WebClient.RequestHeadersSpec.class);
        var responseSpec = mock(org.springframework.web.reactive.function.client.WebClient.ResponseSpec.class);
        var mockWebClient = mock(org.springframework.web.reactive.function.client.WebClient.class);

        when(shinhanApiClient.getClient(anyString())).thenReturn(mockWebClient);
        when(mockWebClient.post()).thenReturn(postSpec);
        when(postSpec.uri(anyString())).thenReturn(postSpec);
        when(postSpec.header(anyString(), anyString())).thenReturn(postSpec);
        // bodyValue()는 RequestHeadersSpec<?>를 반환해야 함
        when(postSpec.bodyValue(any())).thenReturn(headersSpec);
        when(headersSpec.retrieve()).thenReturn(responseSpec);
        when(responseSpec.onStatus(any(), any())).thenReturn(responseSpec);
        when(responseSpec.bodyToMono(EntireExchangeRateResponseDto.class)).thenReturn(reactor.core.publisher.Mono.just(new EntireExchangeRateResponseDto()));

        // when
        RealTimeRateGroupResponseDto response = exchangeRateService.getRealTimeRate();
        // then
        assertNotNull(response, "API response should not be null");
        System.out.println("API Response: " + response);
    }

    @Test
    void testGetLast30DaysRates() throws Exception {
        // given
        String currencyCode = "USD";
        LocalDate endDate = LocalDate.now().minusDays(1);
        LocalDate startDate = endDate.minusDays(29);
        List<ExchangeRate> mockRates = startDate.datesUntil(endDate.plusDays(1))
                .map(date -> ExchangeRate.builder()
                        .id(date.toEpochDay())
                        .currencyCode(currencyCode)
                        .country("United States")
                        .rate(BigDecimal.valueOf(1300 + date.getDayOfMonth()))
                        .unit(1)
                        .baseDate(date)
                        .build())
                .toList();
        when(exchangeRateRepository.findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
                currencyCode, startDate, endDate
        )).thenReturn(mockRates);

        // when
        ExchangeRateHistoriesResponseDto result = exchangeRateService.getExchangeRateHistories();

        // then
        assertNotNull(result);
        assertNotNull(result.getRates());
        assertThat(result.getRates()).hasSize(30);
        assertThat(result.getRates()).allMatch(java.util.Objects::nonNull);
        assertThat(result.getRates()).allMatch(r -> r.getRate() != 0.0);

        // Print as JSON
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        System.out.println("30일간 환율정보: " + json);

        verify(exchangeRateRepository, times(1))
                .findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(currencyCode, startDate, endDate);
    }

    @Test
    void testGetExchangeRatePage() throws Exception {
        // 필요시 ShinhanApiClient, ShinhanApiUtil mock/stub 추가
        // when
        try {
            var result = exchangeRateService.getExchangeRatePage();
            assertNotNull(result, "ExchangeRatePageResponseDto should not be null");
            assertNotNull(result.getExchangeRateHistories(), "exchangeRateHistories should not be null");
            assertNotNull(result.getRealTimeRates(), "latestExchangeRate should not be null");
            assertNotNull(result.getPrediction(), "prediction should not be null");
            assertNotNull(result.getTuition(), "tuition should not be null");
            assertNotNull(result.getExchangeRateHistories().getRates());
            assertThat(result.getExchangeRateHistories().getRates()).allMatch(java.util.Objects::nonNull);
            assertThat(result.getExchangeRateHistories().getRates()).allMatch(r -> r.getRate() != 0.0);
            assertNotNull(result.getRealTimeRates().getUsd());
            assertNotNull(result.getRealTimeRates().getCny());
            assertNotNull(result.getRealTimeRates().getVnd());
            assertThat(result.getRealTimeRates().getUsd().getRate()).isNotEqualTo(0.0);
            assertThat(result.getRealTimeRates().getCny().getRate()).isNotEqualTo(0.0);
            // assertThat(result.getLatestExchangeRate().getVnd().getExchangeRate()).isNotEqualTo(0.0);
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.registerModule(new JavaTimeModule());
            String json = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
            System.out.println("\n===== ExchangeRatePageResponseDto (JSON) =====\n" + json + "\n==============================================\n");
        } catch (com.ssafy.ssashinsa.heyfy.common.exception.CustomException e) {
            System.out.println("CustomException 발생: " + e.getMessage());
        }
    }
}