package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class ShinhanExchangeApiClient {
    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    public ShinhanExchangeResponseDto exchange(String accountNo, String exchangeCurrency, Double exchangeAmount, String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("exchange", "exchange", userKey);
        ShinhanExchangeRequestDto requestDto =
        ShinhanExchangeRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .exchangeCurrency(exchangeCurrency)
                .exchangeAmount(String.valueOf(exchangeAmount.intValue()))
                .build();

        shinhanApiUtil.logRequest(requestDto);
        ShinhanExchangeResponseDto response = apiClient.getClient("exchange")
                .post()
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanExchangeResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }
}
