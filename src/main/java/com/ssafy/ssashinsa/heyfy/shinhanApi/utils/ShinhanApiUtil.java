package com.ssafy.ssashinsa.heyfy.shinhanApi.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomStringUtils;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;

@Slf4j
@Component
@RequiredArgsConstructor
public class ShinhanApiUtil {
    private final ShinhanApiClient shinhanApiClient;
    public ShinhanCommonRequestHeaderDto createHeaderDto(String apiName, String apiServiceCode, String apiKey, String userKey) {
        LocalDateTime now = LocalDateTime.now();
        String transmissionDate = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String transmissionTime = now.format(DateTimeFormatter.ofPattern("HHmmss"));
        String institutionTransactionUniqueNo = now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + RandomStringUtils.randomNumeric(6);
        return ShinhanCommonRequestHeaderDto.builder()
                .apiName(apiName)
                .transmissionDate(transmissionDate)
                .transmissionTime(transmissionTime)
                .fintechAppNo("001")
                .institutionCode("00100")
                .apiServiceCode(apiServiceCode)
                .institutionTransactionUniqueNo(institutionTransactionUniqueNo)
                .apiKey(apiKey)
                .userKey(userKey)
                .build();
    }

    public ShinhanCommonRequestHeaderDto createHeaderDto(String apiName, String apiServiceCode, String userKey) {
        String apiKey = shinhanApiClient.getManagerKey();
        LocalDateTime now = LocalDateTime.now();
        String transmissionDate = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String transmissionTime = now.format(DateTimeFormatter.ofPattern("HHmmss"));
        String institutionTransactionUniqueNo = now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + RandomStringUtils.randomNumeric(6);
        return ShinhanCommonRequestHeaderDto.builder()
                .apiName(apiName)
                .transmissionDate(transmissionDate)
                .transmissionTime(transmissionTime)
                .fintechAppNo("001")
                .institutionCode("00100")
                .apiServiceCode(apiServiceCode)
                .institutionTransactionUniqueNo(institutionTransactionUniqueNo)
                .apiKey(apiKey)
                .userKey(userKey)
                .build();
    }

    public ShinhanCommonRequestHeaderDto createHeaderDto(String apiName, String apiServiceCode) {
        String apiKey = shinhanApiClient.getManagerKey();
        LocalDateTime now = LocalDateTime.now();
        String transmissionDate = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String transmissionTime = now.format(DateTimeFormatter.ofPattern("HHmmss"));
        String institutionTransactionUniqueNo = now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + RandomStringUtils.randomNumeric(6);
        return ShinhanCommonRequestHeaderDto.builder()
                .apiName(apiName)
                .transmissionDate(transmissionDate)
                .transmissionTime(transmissionTime)
                .fintechAppNo("001")
                .institutionCode("00100")
                .apiServiceCode(apiServiceCode)
                .institutionTransactionUniqueNo(institutionTransactionUniqueNo)
                .apiKey(apiKey)
                .build();
    }

    public String getResponseCode(String responseBody) {
        String responseCode = "UNKNOWN";
        try {
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> map = mapper.readValue(responseBody, Map.class);
            responseCode = String.valueOf(map.getOrDefault("responseCode", "UNKNOWN"));
        } catch (Exception e) {
            log.error("Failed to parse responseCode from body: {}", responseBody, e);
        }
        return responseCode;
    }

    // api request, response log
    public void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    public void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }
}
