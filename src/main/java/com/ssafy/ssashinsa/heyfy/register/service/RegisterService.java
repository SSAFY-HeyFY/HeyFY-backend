package com.ssafy.ssashinsa.heyfy.register.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
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
public class RegisterService {

    @Value("${shinhan.manager-key}")
    private String managerKey;

    @Value("${shinhan.account-type-unique-no}")
    private String accountTypeUniqueNo;

    private final UserRepository userRepository;


    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public ShinhanCreateDepositResponseDto createDepositAccount() {
        try {
            String apiKey = managerKey;

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "createDemandDepositAccount",
                    "createDemandDepositAccount",
                    apiKey,
                    userKey
            );

            ShinhanCreateDepositRequestDto requestDto = ShinhanCreateDepositRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountTypeUniqueNo(accountTypeUniqueNo)
                    .build();

            logRequest(requestDto);

            ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/createDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanCreateDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", objectMapper.writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", objectMapper.writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }
}