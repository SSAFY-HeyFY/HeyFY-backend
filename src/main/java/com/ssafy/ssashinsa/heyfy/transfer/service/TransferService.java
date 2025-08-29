package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.ssafy.ssashinsa.heyfy.account.exception.AccountErrorCode;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.RedisUtil;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanForeignDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanException;
import com.ssafy.ssashinsa.heyfy.transfer.exception.TransferErrorCode;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@Slf4j
@RequiredArgsConstructor
public class TransferService {
    private final UserRepository userRepository;
    private final AccountService accountService;
    private final PasswordEncoder passwordEncoder;
    private final ShinhanDemandDepositApiClient shinhanDemandDepositApiClient;
    private final ShinhanForeignDemandDepositApiClient shinhanForeignDemandDepositApiClient;
    private final RedisUtil redisUtil;

    public TransferResponseDto callTransfer(String depositAccountNo, String amount, String transactionSummary, String pinNumber) {
        Users user = findCurrentUser();
        String studentId = user.getStudentId();

        if (redisUtil.isTradePinLocked(studentId)) {
            throw new CustomException(AuthErrorCode.TRADE_LOCKED);
        }

        if (!passwordEncoder.matches(pinNumber, user.getPinNumber())) {
            long failedAttempts = redisUtil.incrementTradePinFailedAttempts(studentId);

            if (failedAttempts >= 5) {
                redisUtil.setTradePinLock(studentId);
                redisUtil.deleteTradePinFailedAttempts(studentId);
                throw new CustomException(AuthErrorCode.PIN_TRADE_ATTEMPTS_EXCEEDED);
            }

            throw new CustomException(AuthErrorCode.INVALID_PIN_NUMBER);
        }

        redisUtil.deleteTradePinFailedAttempts(studentId);
        redisUtil.deleteTradePinLock(studentId);

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getAccount()
                .getAccountNo();
        try {
            return shinhanDemandDepositApiClient.updateDemandDepositAccountTransfer(
                    user.getUserKey(),
                    withdrawalAccountNo,
                    depositAccountNo, amount,
                    transactionSummary,
                    transactionSummary
            );
        } catch (ShinhanException e) {
            ShinhanErrorCode errorCode = e.getErrorCode();
            //1003 1011 1014 1018
            if (errorCode == ShinhanErrorCode.A1014) {
                throw new CustomException(TransferErrorCode.INSUFFICIENT_BALANCE);
            } else if (errorCode == ShinhanErrorCode.A1011) {
                throw new CustomException(TransferErrorCode.INVALID_TRANSACTION_AMOUNT);
            } else if (errorCode == ShinhanErrorCode.A1003) {
                throw new CustomException(TransferErrorCode.INVALID_ACCOUNT_NUMBER);
            } else if (errorCode == ShinhanErrorCode.A1018) {
                throw new CustomException(TransferErrorCode.EXCEEDED_TRANSACTION_SUMMARY_LENGTH);
            } else {
                throw e;
            }
        }
    }

    public TransferResponseDto callForeignTransfer(String depositAccountNo, String amount, String transactionSummary, String pinNumber) {
        Users user = findCurrentUser();

        String studentId = user.getStudentId();

        if (redisUtil.isTradePinLocked(studentId)) {
            throw new CustomException(AuthErrorCode.TRADE_LOCKED);
        }

        if (!passwordEncoder.matches(pinNumber, user.getPinNumber())) {
            long failedAttempts = redisUtil.incrementTradePinFailedAttempts(studentId);

            if (failedAttempts >= 5) {
                redisUtil.setTradePinLock(studentId);
                redisUtil.deleteTradePinFailedAttempts(studentId);
                throw new CustomException(AuthErrorCode.PIN_TRADE_ATTEMPTS_EXCEEDED);
            }

            throw new CustomException(AuthErrorCode.INVALID_PIN_NUMBER);
        }

        redisUtil.deleteTradePinFailedAttempts(studentId);
        redisUtil.deleteTradePinLock(studentId);

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getForeignAccount()
                .getAccountNo();

        try {
            return shinhanForeignDemandDepositApiClient.updateForeignCurrencyDemandDepositAccountTransfer(
                    user.getUserKey(),
                    withdrawalAccountNo,
                    depositAccountNo, amount,
                    transactionSummary,
                    transactionSummary);
        } catch (ShinhanException e) {
            ShinhanErrorCode errorCode = e.getErrorCode();
            //1003 1011 1014 1018 A5005(외화)
            if (errorCode == ShinhanErrorCode.A1014) {
                throw new CustomException(TransferErrorCode.INSUFFICIENT_BALANCE);
            } else if (errorCode == ShinhanErrorCode.A1011) {
                throw new CustomException(TransferErrorCode.INVALID_TRANSACTION_AMOUNT);
            } else if (errorCode == ShinhanErrorCode.A1003) {
                throw new CustomException(TransferErrorCode.INVALID_ACCOUNT_NUMBER);
            } else if (errorCode == ShinhanErrorCode.A1018) {
                throw new CustomException(TransferErrorCode.EXCEEDED_TRANSACTION_SUMMARY_LENGTH);
            } else if (errorCode == ShinhanErrorCode.A5005) {
                throw new CustomException(TransferErrorCode.ONLY_FOREIGN_CURRENCY_ACCOUNT);
            } else {
                throw e;
            }
        }
    }



    private Users findCurrentUser() {
        String studentId = SecurityUtil.getCurrentStudentId();
        if (studentId == null || studentId.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND);
        }
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));
        if (user.getUserKey() == null || user.getUserKey().isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }
        return user;
    }
}
