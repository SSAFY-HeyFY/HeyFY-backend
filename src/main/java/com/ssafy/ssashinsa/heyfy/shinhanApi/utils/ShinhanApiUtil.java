package com.ssafy.ssashinsa.heyfy.shinhanApi.utils;

import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import org.apache.commons.lang3.RandomStringUtils;
import org.springframework.stereotype.Component;

@Component
public class ShinhanApiUtil {
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
}
