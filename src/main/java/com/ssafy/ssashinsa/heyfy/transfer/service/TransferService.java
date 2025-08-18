package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.ssafy.ssashinsa.heyfy.shinhanApi.service.ShinhanApiService;
import com.ssafy.ssashinsa.heyfy.transfer.config.SsafyFinApiProperties;
import com.ssafy.ssashinsa.heyfy.transfer.dto.FinHeader;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferRequestBody;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferResponseBody;
import com.ssafy.ssashinsa.heyfy.transfer.external.SsafyFinApiClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Service
@RequiredArgsConstructor
public class TransferService {

    private final SsafyFinApiClient finClient;
    private final SsafyFinApiProperties apiProperties;
    private final ShinhanApiService shinhanApiService;

    // TODO: 추후 DB에서 사용자 정보를 조회하여 userKey를 가져오도록 수정
    private static final String TEMP_USER_KEY = "37c844c5-9b24-4daa-becb-ca52763a7b39";

    private static final String INSTITUTION_CODE = "00100";
    private static final String FINTECH_APP_NO  = "001";
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyyMMdd");
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HHmmss");

    public TransferResponseBody callTransfer(String withdrawalAccountNo, String depositAccountNo, long amount) {

        LocalDateTime now = LocalDateTime.now();
        String transmissionDate = now.format(DATE_FORMATTER);
        String transmissionTime = now.format(TIME_FORMATTER);

        SecureRandom random = new SecureRandom();
        int sixDigitNumber = random.nextInt(900000) + 100000;
        String uniqueSequence = String.valueOf(sixDigitNumber);
        String institutionTransactionUniqueNo = transmissionDate + transmissionTime + uniqueSequence;

        FinHeader header = FinHeader.builder()
                .apiName("updateDemandDepositAccountTransfer")
                .transmissionDate(transmissionDate)
                .transmissionTime(transmissionTime)
                .institutionCode(INSTITUTION_CODE)
                .fintechAppNo(FINTECH_APP_NO)
                .apiServiceCode("updateDemandDepositAccountTransfer")
                .institutionTransactionUniqueNo(institutionTransactionUniqueNo)
                .apiKey(apiProperties.apiKey())
                .userKey(TEMP_USER_KEY)
                .build();

        TransferRequestBody body = TransferRequestBody.builder()
                .Header(header)
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(String.valueOf(amount))
                .withdrawalTransactionSummary("(수시입출금) : 출금(이체)")
                .depositTransactionSummary("(수시입출금) : 입금(이체)")
                .build();

        return finClient.transfer(body);
    }
}