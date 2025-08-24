package com.ssafy.ssashinsa.heyfy.shinhanApi.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanErrorResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member.ShinhanUserRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member.ShinhanUserResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.IOException;

@Service
@RequiredArgsConstructor
public class ShinhanApiService {
    private static final Logger log = LoggerFactory.getLogger(ShinhanApiService.class);

    private final WebClient webClient;
    private final ShinhanApiClient shinhanApiClient;

    public ShinhanUserResponseDto signUp(String email) {

        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(shinhanApiClient.getManagerKey(), email);

        try {
            return webClient.post()
                    .uri("https://finopenapi.ssafy.io/ssafy/api/v1/member")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, response ->
                            response.bodyToMono(String.class)
                                    .flatMap(body -> {
                                        log.debug("신한은행 API 에러 응답: " + body);

                                        ObjectMapper mapper = new ObjectMapper();
                                        ShinhanErrorResponseDto errorDto;
                                        try {
                                            errorDto = mapper.readValue(body, ShinhanErrorResponseDto.class);
                                        } catch (IOException e) {
                                            return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                                        }

                                        if ("E4002".equals(errorDto.getResponseCode())) {
                                            return Mono.error(new CustomException(ShinhanApiErrorCode.API_USER_ALREADY_EXISTS));
                                        }
                                        return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                                    }))
                    .bodyToMono(ShinhanUserResponseDto.class)
                    .block();
        } catch (CustomException e) {

            // =======================================================
            if (e.getErrorCode() == ShinhanApiErrorCode.API_USER_ALREADY_EXISTS) {
                log.debug("이미 존재하는 유저. 유저 조회 API 요청");
                return searchUser(email);
            }
            // =======================================================

            throw e;
        }
    }

    public ShinhanUserResponseDto searchUser(String email) {

        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(shinhanApiClient.getManagerKey(), email);

        return webClient.post()
                .uri("https://finopenapi.ssafy.io/ssafy/api/v1/member/search")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, response ->
                        response.bodyToMono(String.class)
                                .flatMap(body -> {
                                    log.debug("신한은행 API 에러 응답 (유저 조회): " + body);

                                    return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                                })
                )
                .bodyToMono(ShinhanUserResponseDto.class)
                .block();
    }
}