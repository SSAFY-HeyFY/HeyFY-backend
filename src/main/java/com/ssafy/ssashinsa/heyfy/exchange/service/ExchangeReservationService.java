package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CommonErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.RedisUtil;
import com.ssafy.ssashinsa.heyfy.exchange.domain.Currency;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.exception.ExchangeErrorCode;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeReservationRepository;
import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRealTimeRatesDto;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanExchangeApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanForeignDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ExchangeReservationService {

    private final ExchangeReservationRepository exchangeReservationRepository;
    private final UserRepository userRepository;
    private final FastApiClient fastApiClient;
    private final ShinhanExchangeApiClient shinhanExchangeApiClient;
    private final ShinhanForeignDemandDepositApiClient shinhanForeignDemandDepositApiClient;
    private final RedisUtil redisUtil;
    private final ShinhanDemandDepositApiClient shinhanDemandDepositApiClient;

    @Transactional
    public ExchangeReservation cancelExchangeReservation(String studentId, Long reservationId, String pinNumber) {
        Users user = userRepository.findUserWithAccountsByStudentId(studentId)
                .orElseThrow(() -> new CustomException(CommonErrorCode.USER_NOT_FOUND, "사용자를 찾을 수 없습니다: " + studentId));

        redisUtil.validateTradePin(studentId, pinNumber, user.getPinNumber());
        ExchangeReservation reservation = exchangeReservationRepository.findByIdWithUser(reservationId);
        if (!reservation.getUser().getStudentId().equals(studentId)) {
            throw new CustomException(AuthErrorCode.UNAUTHORIZED, "권한이 없습니다: " + studentId);
        }
        if (reservation.isCanceled()) {
            throw new CustomException(ExchangeErrorCode.ALREADY_CANCELED);
        }
        if (reservation.isExchangeCompleted()) {
            throw new CustomException(ExchangeErrorCode.ALREADY_COMPLETED);
        }
        reservation.cancel();
        log.info("환전 예약 취소: " + reservation);
        return reservation;
    }

    @Transactional
    public List<ExchangeReservation> getExchangeReservations(String studentId) {
        List<ExchangeReservation> reservations = exchangeReservationRepository.findByStudentId(studentId);
        log.info("환전 예약 조회: " + reservations.size() + "건");
        return reservations;
    }


    @Transactional
    public ExchangeReservation createExchangeReservation(String studentId, @RequestBody ExchangeReservationRequestDto requestDto) {
        Users user = userRepository.findUserWithAccountsByStudentId(studentId)
                .orElseThrow(() -> new CustomException(CommonErrorCode.USER_NOT_FOUND, "사용자를 찾을 수 없습니다: " + studentId));
        log.info("유저 조회: " + user.getName());

        String pinNumber = requestDto.getPinNumber();
        redisUtil.validateTradePin(studentId, pinNumber, user.getPinNumber());

        ForeignAccount foreignAccount = user.getForeignAccount();
        Account account = user.getAccount();

        String withdrawalAccountNo = "";
        String depositAccountNo = "";
        Currency withdrawalAccountCurrency = Currency.USD;
        Currency depositAccountCurrency = Currency.KRW;
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
                requestDto.getTransactionBalance(),
                requestDto.getBaseExchangeRate()
        );

        log.info("환전 예약 생성: " + reservation);
        return exchangeReservationRepository.save(reservation);
    }

    @Transactional
    public List<FcmToken> exchangeToForeignBatch() {
        log.info("환전 예약 배치 작업 시작");
        FastApiRealTimeRatesDto realTimeRates = fastApiClient.getRealTimeRates();
        Double usdRate = realTimeRates.getData().stream()
                .filter(rate -> "USDKRW".equals(rate.getCurrency()))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("USD 환율 데이터가 없습니다."))
                .getRate();
        log.info("현재 USD 환율: " + usdRate);
        List<ExchangeReservation> reservationList = exchangeReservationRepository.findAllNotCanceledAndNotCompletedWithUser();
        log.info("처리 대상 예약 수: " + reservationList.size());
        List<ExchangeReservation> exchangeList = reservationList.stream()
                .filter(reservation -> reservation.getBaseExchangeRate() < usdRate)
                .collect(Collectors.toList());
        log.info("환전 처리 대상 예약 수: " + exchangeList.size());
        List<FcmToken> result = exchangeList.stream()
                .map(reservation -> {
                    try {
                        log.info("환전 처리 시작: reservationId=" + reservation.getId());
                        shinhanExchangeApiClient.exchange(
                                reservation.getWithdrawalAccountNo(),
                                reservation.getDepositAccountCurrency().toString(),
                                reservation.getTransactionBalance(),
                                reservation.getUser().getUserKey()
                        );
                        log.info("환전 처리 완료, 한화 입금 시작: reservationId=" + reservation.getId());
                        shinhanDemandDepositApiClient.updateDemandDepositAccountDeposit(
                                reservation.getDepositAccountNo(),
                                reservation.getTransactionBalance(),
                                reservation.getUser().getUserKey()
                        );
                        log.info("한화 입금 완료: reservationId=" + reservation.getId());
                        return reservation.getUser().getFcmTokens().get(0);
                    } catch (Exception e) {
                        log.error("환전 처리 중 오류 발생: reservationId={}, error={}", reservation.getId(), e.getMessage(), e);
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        List<Long> ids = exchangeList.stream().map(ExchangeReservation::getId).collect(Collectors.toList());
        exchangeReservationRepository.bulkComplete(ids);
        return result;
    }
}
