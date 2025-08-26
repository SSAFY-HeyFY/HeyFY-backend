package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthCheckRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthCheckResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthResponseDto;
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
public class ShinhanAccountAuthApiClient {
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    public AccountAuthResponseDto openAccountAuth(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("openAccountAuth", "openAccountAuth", userKey);

        AccountAuthRequestDto requestDto = AccountAuthRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .authText("SSAFY")
                .build();

        shinhanApiUtil.logRequest(requestDto);

        AccountAuthResponseDto response = shinhanApiClient.getClient("account-auth")
                .post()
                .uri("/openAccountAuth")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(AccountAuthResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public AccountAuthCheckResponseDto checkAuthCode(String userKey, String accountNo, String authCode) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("checkAuthCode", "checkAuthCode", userKey);

        AccountAuthCheckRequestDto requestDto = AccountAuthCheckRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .authText("SSAFY")
                .authCode(authCode)
                .build();

        shinhanApiUtil.logRequest(requestDto);

        AccountAuthCheckResponseDto response = shinhanApiClient.getClient("account-auth")
                .post()
                .uri("/checkAuthCode")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(AccountAuthCheckResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }
}
