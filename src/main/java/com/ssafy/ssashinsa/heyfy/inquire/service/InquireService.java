package com.ssafy.ssashinsa.heyfy.inquire.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.inquire.exception.ShinhanInquireApiErrorCode;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
@RequiredArgsConstructor
public class InquireService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ShinhanDemandDepositApiClient shinhanDemandDepositApiClient;

    public boolean checkAccount() {
        String studentId = SecurityUtil.getCurrentStudentId();
        if (studentId == null) {
            return false;
        }

        Users user = userRepository.findByStudentId(studentId).orElse(null);

        if (user != null && accountRepository.findByUser(user).isPresent()) {
            return true;
        }
        return false;
    }

    public ShinhanInquireSingleDepositResponseDto inquireSingleDeposit() {

        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
        }

        String accountNo = accountRepository.findByUser(user)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND))
                .getAccountNo();

        ShinhanInquireSingleDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandDepositAccount(userKey, accountNo);

        return response;
    }

    public ShinhanInquireSingleDepositResponseDto inquireSingleDeposit(String accountNo) {

        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
        }

        Account account = accountRepository.findByUserAndAccountNo(user, accountNo)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

        ShinhanInquireSingleDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandDepositAccount(userKey, accountNo);

        return response;
    }



    public ShinhanInquireSingleDepositResponseDto inquireSingleForeignDeposit() {
        try {
            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
            }

            String accountNo = foreignAccountRepository.findByUser(user)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND))
                    .getAccountNo();

            ShinhanInquireSingleDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandForeignDepositAccount(userKey, accountNo);

            return response;
        } catch (Exception e) {
            log.error("계좌 등록 API 호출 실패 : {}", e.getMessage(), e);
            throw e;
        }
    }

    public ShinhanInquireSingleDepositResponseDto inquireSingleForeignDeposit(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
        }
//            ForeignAccount account = foreignAccountRepository.findByUserAndAccountNo(user, accountNo)
//                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

        ShinhanInquireSingleDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandForeignDepositAccount(userKey, accountNo);

        return response;
    }


    public ShinhanInquireDepositResponseDto inquireDepositList() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
        }
        ShinhanInquireDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandDepositAccountList(userKey);

        return response;
    }


    public InquireTransactionHistoryResponseDto getTransactionHistory() {
        String apiKey = shinhanApiClient.getManagerKey();

        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        String accountNo = accountRepository.findByUser(user)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND))
                .getAccountNo();

        InquireTransactionHistoryResponseDto response = shinhanDemandDepositApiClient.inquireTransactionHistoryList(userKey, accountNo);

        ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                "inquireTransactionHistoryList",
                "inquireTransactionHistoryList",
                apiKey,
                userKey
        );

        return response;
    }

    public InquireTransactionHistoryResponseDto getTransactionHistory(String accountNo) {
        String apiKey = shinhanApiClient.getManagerKey();

        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

//            Account account = accountRepository.findByUserAndAccountNo(user, accountNo)
//                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

        InquireTransactionHistoryResponseDto response = shinhanDemandDepositApiClient.inquireTransactionHistoryList(userKey, accountNo);

        return response;
    }

    public InquireSingleTransactionHistoryResponseDto getSingleTransactionHistory(String accountNo, String transactionUniqueNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        InquireSingleTransactionHistoryResponseDto response = shinhanDemandDepositApiClient.inquireTransactionHistory(userKey, accountNo, transactionUniqueNo);

        return response;
    }

    public InquireTransactionHistoryResponseDto getForeignTransactionHistory(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        InquireTransactionHistoryResponseDto response = shinhanDemandDepositApiClient.inquireForeignTransactionHistoryList(userKey, accountNo);

        return response;
    }
}
