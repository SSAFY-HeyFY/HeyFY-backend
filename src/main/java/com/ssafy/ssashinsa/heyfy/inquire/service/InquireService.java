package com.ssafy.ssashinsa.heyfy.inquire.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ExchangeHistorySimplifiedDto;
import com.ssafy.ssashinsa.heyfy.inquire.exception.ShinhanInquireApiErrorCode;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.ExchangeHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

import java.math.RoundingMode;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Service
@Slf4j
@RequiredArgsConstructor
public class InquireService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanDemandDepositApiClient shinhanDemandDepositApiClient;

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

        return shinhanDemandDepositApiClient.inquireDemandDepositAccount(userKey, accountNo);
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

        return shinhanDemandDepositApiClient.inquireDemandDepositAccount(userKey, accountNo);
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

        return shinhanDemandDepositApiClient.inquireDemandForeignDepositAccount(userKey, accountNo);
    }


    public ShinhanInquireDepositResponseDto inquireDepositList() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanInquireApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanInquireApiErrorCode.MISSING_USER_KEY);
        }

        return shinhanDemandDepositApiClient.inquireDemandDepositAccountList(userKey);
    }

    public InquireTransactionHistoryResponseDto getTransactionHistory(String accountNo) {

        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

//            Account account = accountRepository.findByUserAndAccountNo(user, accountNo)
//                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

        return shinhanDemandDepositApiClient.inquireTransactionHistoryList(userKey, accountNo);
    }

    public InquireSingleTransactionHistoryResponseDto getSingleTransactionHistory(String accountNo, String transactionUniqueNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        return shinhanDemandDepositApiClient.inquireTransactionHistory(userKey, accountNo, transactionUniqueNo);
    }

    public InquireTransactionHistoryResponseDto getForeignTransactionHistory(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        return shinhanDemandDepositApiClient.inquireForeignTransactionHistoryList(userKey, accountNo);
    }

    public ExchangeHistoryResponseDto getExchangeHistory(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));


        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        return shinhanDemandDepositApiClient.exchangeHistory(userKey, accountNo);
    }

    public List<ExchangeHistorySimplifiedDto> getSimplifiedExchangeHistory() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        Account account = accountRepository.findByUser(user)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));
        String accountNo = account.getAccountNo();

        ForeignAccount foreignAccount = foreignAccountRepository.findByUser(user)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));
        String foreignAccountNo = foreignAccount.getAccountNo();

        ExchangeHistoryResponseDto originalResponseFromAccount = this.getExchangeHistory(accountNo);
        ExchangeHistoryResponseDto originalResponseFromForeignAccount = this.getExchangeHistory(foreignAccountNo);

        Stream<ExchangeHistorySimplifiedDto> firstListStream = Stream.empty();
        if (originalResponseFromAccount != null && originalResponseFromAccount.getREC() != null) {
            firstListStream = originalResponseFromAccount.getREC().stream()
                    .map(recDto -> {
                        String currencyAmount = Optional.ofNullable(recDto.getCurrency().getAmount())
                                .map(BigDecimal::new)
                                .map(bd -> bd.setScale(1, RoundingMode.HALF_UP))
                                .map(BigDecimal::toPlainString)
                                .orElse(null);

                        String exchangeAmount = Optional.ofNullable(recDto.getExchangeCurrency().getAmount())
                                .map(BigDecimal::new)
                                .map(bd -> {
                                    if ("KRW".equals(recDto.getExchangeCurrency().getCurrency())) {
                                        return bd.stripTrailingZeros();
                                    } else {
                                        return bd.setScale(1, RoundingMode.HALF_UP);
                                    }
                                })
                                .map(BigDecimal::toPlainString)
                                .orElse(null);

                        String exchangeRate = Optional.ofNullable(recDto.getExchangeCurrency().getExchangeRate())
                                .map(Object::toString)
                                .map(s -> s.replace(",", ""))
                                .orElse(null);

                        return ExchangeHistorySimplifiedDto.builder()
                                .fromAccountNo(accountNo)
                                .toAccountNo(foreignAccountNo)
                                .currency(recDto.getCurrency().getCurrency())
                                .currencyName(recDto.getCurrency().getCurrencyName())
                                .amount(currencyAmount)
                                .exchangeCurrency(recDto.getExchangeCurrency().getCurrency())
                                .exchangeCurrencyName(recDto.getExchangeCurrency().getCurrencyName())
                                .exchangeAmount(exchangeAmount)
                                .exchangeRate(exchangeRate)
                                .created(recDto.getCreated())
                                .build();
                    });
        }

        Stream<ExchangeHistorySimplifiedDto> secondListStream = Stream.empty();
        if (originalResponseFromForeignAccount != null && originalResponseFromForeignAccount.getREC() != null) {
            secondListStream = originalResponseFromForeignAccount.getREC().stream()
                    .map(recDto -> {
                        String currencyAmount = Optional.ofNullable(recDto.getCurrency().getAmount())
                                .map(BigDecimal::new)
                                .map(bd -> bd.setScale(1, RoundingMode.HALF_UP))
                                .map(BigDecimal::toPlainString)
                                .orElse(null);

                        String exchangeAmount = Optional.ofNullable(recDto.getExchangeCurrency().getAmount())
                                .map(BigDecimal::new)
                                .map(bd -> {
                                    if ("KRW".equals(recDto.getExchangeCurrency().getCurrency())) {
                                        return bd.stripTrailingZeros();
                                    } else {
                                        return bd.setScale(1, RoundingMode.HALF_UP);
                                    }
                                })
                                .map(BigDecimal::toPlainString)
                                .orElse(null);

                        BigDecimal calculatedRate = Optional.ofNullable(recDto.getExchangeCurrency().getAmount())
                                .flatMap(exAmount -> Optional.ofNullable(recDto.getCurrency().getAmount())
                                        .map(amount -> BigDecimal.valueOf(exAmount).divide(BigDecimal.valueOf(amount), 1, RoundingMode.HALF_UP)))
                                .orElse(null);

                        String exchangeRate = (calculatedRate != null) ? calculatedRate.toPlainString() : null;

                        return ExchangeHistorySimplifiedDto.builder()
                                .fromAccountNo(foreignAccountNo)
                                .toAccountNo(accountNo)
                                .currency(recDto.getCurrency().getCurrency())
                                .currencyName(recDto.getCurrency().getCurrencyName())
                                .amount(currencyAmount)
                                .exchangeCurrency(recDto.getExchangeCurrency().getCurrency())
                                .exchangeCurrencyName(recDto.getExchangeCurrency().getCurrencyName())
                                .exchangeAmount(exchangeAmount)
                                .exchangeRate(exchangeRate)
                                .created(recDto.getCreated())
                                .build();
                    });
        }

        return Stream.concat(firstListStream, secondListStream)
                .sorted(Comparator.comparing(ExchangeHistorySimplifiedDto::getCreated, Comparator.reverseOrder()))
                .collect(Collectors.toList());
    }
}