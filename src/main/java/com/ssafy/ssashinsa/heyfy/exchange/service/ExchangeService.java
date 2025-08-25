package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.exception.ExchangeErrorCode;
import com.ssafy.ssashinsa.heyfy.exchange.util.ShinhanExchangeClient;
import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRateAnalysisDto;
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

    private final FastApiClient fastApiClient;
    private final ShinhanExchangeClient shinhanExchangeiClient;
    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;

    @Transactional
    public ExchangeResponseDto exchangeToForeign(String studentId, ExchangeRequestDto exchangeRequestDto) {
        Users user = userRepository.findUserWithAccountsByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = user.getAccount();
        ForeignAccount foreignAccount = user.getForeignAccount();
        // account, foreignAccount 둘 다 존재해야 함
        if (account == null || foreignAccount == null) {
            throw new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND);
        }

        // 환전 api 호출
        ShinhanExchangeResponseDto exchangeResponse = shinhanExchangeiClient.exchange(
                account.getAccountNo(), "USD", exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        // 환전 금액만큼 계좌에 입금
        ShinhanUpdateAccountResponseDto updateAccountResponse
                = shinhanExchangeiClient.updateForeignAccount(
                foreignAccount.getAccountNo(), exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        // 입금된 계좌 잔액 조회
        ShinhanInquireDemandDepositAccountBalanceResponseDto foreignAccountBalanceResponse
                = shinhanExchangeiClient.getForeignAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());

        // 출금 계좌 잔액
        Double depositAccountBalance = exchangeResponse.getREC().getAccountInfo().getBalance();
        // 입금 계좌 잔액
        Double withdrawalAccountBalance = foreignAccountBalanceResponse.getREC().getAccountBalance();

        return ExchangeResponseDto.builder()
                .depositAccountBalance(depositAccountBalance)
                .withdrawalAccountBalance(withdrawalAccountBalance)
                .transactionBalance(exchangeRequestDto.getTransactionBalance())
                .build();
    }

    @Transactional
    public ExchangeResponseDto exchangeFromForeign(String studentId, ExchangeRequestDto exchangeRequestDto) {
        Users user = userRepository.findUserWithAccountsByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = user.getAccount();
        ForeignAccount foreignAccount = user.getForeignAccount();
        // account, foreignAccount 둘 다 존재해야 함
        if (account == null || foreignAccount == null) {
            throw new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND);
        }

        // 환전 api 호출
        ShinhanExchangeResponseDto exchangeResponse = shinhanExchangeiClient.exchange(
                account.getAccountNo(), "KRW", exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        // 환전 금액만큼 계좌에 입금
        ShinhanUpdateAccountResponseDto updateAccountResponse = shinhanExchangeiClient.updateAccount(
                account.getAccountNo(),exchangeRequestDto.getTransactionBalance(), user.getUserKey());

        // 입금된 계좌 잔액 조회
        ShinhanInquireDemandDepositAccountBalanceResponseDto accountBalanceResponse
                = shinhanExchangeiClient.getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());

        // 출금 계좌 잔액
        Double depositAccountBalance = exchangeResponse.getREC().getAccountInfo().getBalance();
        // 입금 계좌 잔액
        Double withdrawalAccountBalance = accountBalanceResponse.getREC().getAccountBalance();

        return ExchangeResponseDto.builder()
                .depositAccountBalance(depositAccountBalance)
                .withdrawalAccountBalance(withdrawalAccountBalance)
                .transactionBalance(exchangeRequestDto.getTransactionBalance())
                .build();
    }

    @Transactional
    public AccountBalanceResponseDto getAccountBalance(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));


        ShinhanInquireDemandDepositAccountBalanceResponseDto response = shinhanExchangeiClient.getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());

        return AccountBalanceResponseDto.builder()
                .accountNo(response.getREC().getAccountNo())
                .accountBalance(response.getREC().getAccountBalance())
                .currency(response.getREC().getCurrency())
                .build();
    }

    @Transactional
    public AccountBalanceResponseDto getForeignAccountBalance(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        ForeignAccount foreignAccount = foreignAccountRepository.findForeignAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));


        ShinhanInquireDemandDepositAccountBalanceResponseDto response = shinhanExchangeiClient.getForeignAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());

        return AccountBalanceResponseDto.builder()
                .accountNo(foreignAccount.getAccountNo())
                .accountBalance(response.getREC().getAccountBalance())
                .currency(response.getREC().getCurrency())
                .isForeign(true)
                .build();
    }

    @Transactional
    public HistoricalAnalysisResponseDto getHistoricalAnalysis() {
        return HistoricalAnalysisResponseDto.builder()
                .message("Over the past 30 days, today shows the highest exchange rate")
                .build();
    }

    @Transactional
    public AIPredictionResponseDto getExchangeRateAIPrediction() {
        return AIPredictionResponseDto.builder()
                .message("The rate may increase by $0.54 more in the near future")
                .build();
    }

    @Transactional
    public RateAnalysisResponseDto getRateAnalysis() {
        FastApiRateAnalysisDto apiResponse = fastApiClient.getRateAnalysis();
        return RateAnalysisResponseDto.builder()
                .historicalAnalysis(HistoricalAnalysisResponseDto.builder()
                        .message("Over the past 30 days, today shows the highest exchange rate")
                        .build())
                .aiPrediction(AIPredictionResponseDto.builder()
                        .message("The rate may increase by $0.54 more in the near future")
                        .build())
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
                = shinhanExchangeiClient.getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto accountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(accountBalance.getREC().getAccountNo())
                .accountBalance(accountBalance.getREC().getAccountBalance())
                .currency(accountBalance.getREC().getCurrency())
                .isForeign(false)
                .build();

        // foreign account balance
        ShinhanInquireDemandDepositAccountBalanceResponseDto foreignAccountBalance
                = shinhanExchangeiClient.getForeignAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto foreignAccountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(foreignAccountBalance.getREC().getAccountNo())
                .accountBalance(foreignAccountBalance.getREC().getAccountBalance())
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
