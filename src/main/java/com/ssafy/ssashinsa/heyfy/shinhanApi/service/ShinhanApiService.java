package com.ssafy.ssashinsa.heyfy.shinhanApi.service;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanErrorResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanUserRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanUserResponseDto;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class ShinhanApiService {
    private final WebClient webClient;

    @Value("${shinhan.manager-key}")
    private String mangerKey;

    public ShinhanUserResponseDto signUp(String email) {

        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(mangerKey, email);

        try {
            return webClient.post()
                    .uri("https://finopenapi.ssafy.io/ssafy/api/v1/member")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, response ->
                            response.bodyToMono(ShinhanErrorResponseDto.class)
                                    .flatMap(errorDto -> {
                                        if ("E4002".equals(errorDto.getResponseCode())) {
                                            return Mono.error(new CustomException(ShinhanApiErrorCode.API_USER_ALREADY_EXISTS));
                                        }
                                        return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                                    }))
                    .bodyToMono(ShinhanUserResponseDto.class)
                    .block();
        } catch (CustomException e) {

            // =======================================================
            // 신한은행 API 에러코드 E4002일 경우, 유저 조회 로직
            // 주석 처리된 부분은 후에 보안 로직으로 지적받을 시 주석처리하거나 삭제
            if (e.getErrorCode() == ShinhanApiErrorCode.API_USER_ALREADY_EXISTS) {
                System.out.println("이미 존재하는 유저. 유저 조회 API 요청");
                return searchUser(email);
            }
            // =======================================================

            throw e;
        }
    }

    public ShinhanUserResponseDto searchUser(String email) {

        ShinhanUserRequestDto requestDto = new ShinhanUserRequestDto(mangerKey, email);

        return webClient.post()
                .uri("https://finopenapi.ssafy.io/ssafy/api/v1/member/search")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, response ->
                        response.bodyToMono(ShinhanErrorResponseDto.class)
                                .flatMap(errorDto -> {
                                    throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
                                })
                )
                .bodyToMono(ShinhanUserResponseDto.class)
                .block();
    }
}


