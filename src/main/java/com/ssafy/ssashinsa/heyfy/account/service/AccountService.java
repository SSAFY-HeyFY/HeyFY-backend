package com.ssafy.ssashinsa.heyfy.account.service;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountPairDto;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanAccountAuthApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthCheckResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@Slf4j
@RequiredArgsConstructor
public class AccountService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ShinhanAccountAuthApiClient shinhanAccountAuthApiClient;

    public Optional<AccountPairDto> getAccounts() {
        String studentId = SecurityUtil.getCurrentStudentId();
//        String userEmail = userRepository.findByStudentId(studentId)
//                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND))
//                .getEmail();
        String userEmail = SecurityUtil.getCurrentUserEmail();
        return userRepository.findAccountsByUserEmail(userEmail);
    }

    public AccountAuthResponseDto AccountAuth() {
        try {
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

            AccountAuthResponseDto response = shinhanAccountAuthApiClient.openAccountAuth(userKey, accountNo);

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public AccountAuthResponseDto AccountAuth(String accountNo) {
        try {
            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            AccountAuthResponseDto response = shinhanAccountAuthApiClient.openAccountAuth(userKey, accountNo);

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public AccountAuthCheckResponseDto accountAuthCheck(String accountNo, String authCode) {
        try {
            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            AccountAuthCheckResponseDto response = shinhanAccountAuthApiClient.checkAuthCode(userKey, accountNo, authCode);

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

}