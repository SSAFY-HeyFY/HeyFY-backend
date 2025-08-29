package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.common.exception.CommonErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.domain.Currency;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeReservationRepository;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ExchangeReservationService {

    private final ExchangeReservationRepository exchangeReservationRepository;
    private final UserRepository userRepository;

    @Transactional
    public ExchangeReservation createExchangeReservation(String studentId, ExchangeReservationRequestDto requestDto) {

        Users user = userRepository.findUserWithAccountsByStudentId(studentId)
                .orElseThrow(() -> new CustomException(CommonErrorCode.USER_NOT_FOUND, "사용자를 찾을 수 없습니다: " + studentId));

        ForeignAccount foreignAccount = user.getForeignAccount();
        Account account = user.getAccount();

        String withdrawalAccountNo = "";
        String depositAccountNo = "";
        Currency withdrawalAccountCurrency = Currency.KRW;
        Currency depositAccountCurrency = Currency.USD;
        if (requestDto.getCurrency().equals("KRW")) {
            withdrawalAccountNo = foreignAccount.getAccountNo();
            withdrawalAccountCurrency = Currency.USD;
            depositAccountNo = account.getAccountNo();
            depositAccountCurrency = Currency.KRW;
        }
        if (withdrawalAccountNo.equals("") || depositAccountNo.equals("")) {
            throw new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND, "Account not found.");
        }

        ExchangeReservation reservation = ExchangeReservation.create(
                user,
                withdrawalAccountNo, withdrawalAccountCurrency,
                depositAccountNo, depositAccountCurrency,
                requestDto.getTransactionBalance()
                );


        return exchangeReservationRepository.save(reservation);
    }
}
