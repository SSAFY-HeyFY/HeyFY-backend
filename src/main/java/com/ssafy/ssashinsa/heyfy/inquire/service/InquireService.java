package com.ssafy.ssashinsa.heyfy.inquire.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.exception.ShinhanInquireApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class InquireService {

    @Value("${shinhan.manager-key}")
    private String managerKey;

    private final UserRepository userRepository;


    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public boolean checkAccount() {
        ShinhanInquireDepositResponseDto response = inquireDepositResponseDto();

        if (response.getREC() == null || response.getREC().isEmpty()) {
            return false;
        }
        return true;
    }


    public ShinhanInquireDepositResponseDto inquireDepositResponseDto() {
        try {
            String apiKey = managerKey;

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "inquireDemandDepositAccountList",
                    "inquireDemandDepositAccountList",
                    apiKey,
                    userKey
            );

            ShinhanInquireDepositRequestDto requestDto = ShinhanInquireDepositRequestDto.builder()
                    .Header(commonHeaderDto)
                    .build();

            logRequest(requestDto);

            ShinhanInquireDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/inquireDemandDepositAccountList")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanInquireApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanInquireDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();


            return response;
        } catch (Exception e) {
            log.error("계좌 등록 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }


    private void logRequest(Object requestDto) {
        try {
            log.info("요청 JSON : {}", objectMapper.writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("요청 에러 : ", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("응답 JSON : {}", objectMapper.writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("응답 에러 : ", e);
        }
    }
}
