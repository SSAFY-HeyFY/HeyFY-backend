package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member.ShinhanUserRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member.ShinhanUserResponseDto;
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
public class ShinhanMemberApiClient {
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;


    public ShinhanUserResponseDto registerMember(String email) {
        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(shinhanApiClient.getManagerKey(), email);
        shinhanApiUtil.logRequest(requestDto);

        ShinhanUserResponseDto response = shinhanApiClient.getClient("member")
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
                .bodyToMono(ShinhanUserResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanUserResponseDto getMember(String email) {
        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(shinhanApiClient.getManagerKey(), email);
        shinhanApiUtil.logRequest(requestDto);

        ShinhanUserResponseDto response = shinhanApiClient.getClient("member")
                .post()
                .uri("/search")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanUserResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }
}
