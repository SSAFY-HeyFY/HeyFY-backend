package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

@Slf4j
@Component
@RequiredArgsConstructor
public class ShinhanExchangeRateApiClient {
    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;


    public EntireExchangeRateResponseDto getExchangeRateFromExternalApi() {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("exchangeRate", "exchangeRate");
        ExchangeRateRequestDto requestDto = ExchangeRateRequestDto.builder()
                .Header(header)
                .build();

        shinhanApiUtil.logRequest(requestDto);

        EntireExchangeRateResponseDto response = apiClient.getClient("exchange-rate")
                .post()
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                        }))
                .bodyToMono(EntireExchangeRateResponseDto.class)
                .doOnNext(shinhanApiUtil::logRequest)
                .block();
        shinhanApiUtil.logResponse(response);
        return response;

    }
}
