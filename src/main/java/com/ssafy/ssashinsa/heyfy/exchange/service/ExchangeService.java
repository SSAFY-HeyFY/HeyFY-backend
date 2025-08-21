package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseRecDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.exception.ExchangeErrorCode;
import com.ssafy.ssashinsa.heyfy.exchange.util.ShinhanExchangeApiClient;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


@Slf4j
@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class ExchangeService {

    private final ShinhanExchangeApiClient apiClient;
    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;

    @Transactional
    public ShinhanExchangeResponseRecDto exchangeToForeign(String studentId, ExchangeRequestDto exchangeRequestDto) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));

        ShinhanExchangeResponseDto exchangeResponse = apiClient.exchange(
                account.getAccountNo(), exchangeRequestDto.getWithdrawalAccountCurrency(), exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        ShinhanUpdateAccountResponseDto updateAccountResponse = apiClient.updateForeignAccount(
                exchangeRequestDto.getWithdrawalAccountNo(),exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        return exchangeResponse.getREC();
    }
    @Transactional
    public ShinhanExchangeResponseRecDto exchangeFromForeign(String studentId, ExchangeRequestDto exchangeRequestDto) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        ForeignAccount account = foreignAccountRepository.findForeignAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));

        ShinhanExchangeResponseDto exchangeResponse = apiClient.exchange(
                account.getAccountNo(), exchangeRequestDto.getWithdrawalAccountCurrency(), exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        ShinhanUpdateAccountResponseDto updateAccountResponse = apiClient.updateAccount(
                exchangeRequestDto.getWithdrawalAccountNo(),exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        return exchangeResponse.getREC();
    }

    @Transactional
    public AccountBalanceResponseDto getAccountBalance(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));


        ShinhanInquireDemandDepositAccountBalanceResponseDto response = apiClient.getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());

        return AccountBalanceResponseDto.builder()
                .accountNo(response.getREC().getAccountNo())
                .accountBalance(Integer.parseInt(response.getREC().getAccountBalance()))
                .currency(response.getREC().getCurrency())
                .build();
    }
    @Transactional
    public AccountBalanceResponseDto getForeignAccountBalance(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        ForeignAccount foreignAccount = foreignAccountRepository.findForeignAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));


        ShinhanInquireDemandDepositAccountBalanceResponseDto response = apiClient.getForeignAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());

        return AccountBalanceResponseDto.builder()
                .accountNo(foreignAccount.getAccountNo())
                .accountBalance(Integer.parseInt(response.getREC().getAccountBalance()))
                .currency(response.getREC().getCurrency())
                .isForeign(true)
                .build();
    }

    @Transactional
    public HistoricalAnalysisResponseDto getHistoricalAnalysis(){
        return HistoricalAnalysisResponseDto.builder()
                .message("Over the past 30 days, today shows the highest exchange rate")
                .build();
    }

    @Transactional
    public AIPredictionResponseDto getExchangeRateAIPrediction(){
        return AIPredictionResponseDto.builder()
                .message("The rate may increase by $0.54 more in the near future")
                .build();
    }

    @Transactional
    public ExchangePageResponseDto getExchangePage(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));
        ForeignAccount foreignAccount = foreignAccountRepository.findForeignAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));

        // account balance
        ShinhanInquireDemandDepositAccountBalanceResponseDto accountBalance
                = apiClient.getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto accountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(accountBalance.getREC().getAccountNo())
                .accountBalance(Integer.parseInt(accountBalance.getREC().getAccountBalance()))
                .currency(accountBalance.getREC().getCurrency())
                .isForeign(false)
                .build();

        // foreign account balance
        ShinhanInquireDemandDepositAccountBalanceResponseDto foreignAccountBalance
                = apiClient.getForeignAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto foreignAccountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(foreignAccountBalance.getREC().getAccountNo())
                .accountBalance(Integer.parseInt(foreignAccountBalance.getREC().getAccountBalance()))
                .currency(foreignAccountBalance.getREC().getCurrency())
                .isForeign(true)
                .build();

        // AI Prediction
        AIPredictionResponseDto aiPredictionResponseDto = AIPredictionResponseDto.builder()
                .message("The rate may increase by $0.54 more in the near future")
                .build();
        // Historical Analysis
        HistoricalAnalysisResponseDto historicalAnalysisResponseDto = HistoricalAnalysisResponseDto.builder()
                .message("Over the past 30 days, today shows the highest exchange rate")
                .build();
        return ExchangePageResponseDto.builder()
                .aiPrediction(aiPredictionResponseDto)
                .historicalAnalysis(historicalAnalysisResponseDto)
                .accountBalance(accountBalanceResponseDto)
                .foreignAccountBalance(foreignAccountBalanceResponseDto)
                .build();
    }





}
