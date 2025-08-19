package com.ssafy.ssashinsa.heyfy.exchange.repository;


import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface ExchangeRateRepository extends JpaRepository<ExchangeRate, Long> {
    /**
     * 특정 날짜 + 통화 코드의 환율 조회
     */
    Optional<ExchangeRate> findByBaseDateAndCurrencyCode(LocalDate baseDate, String currencyCode);

    /**
     * 특정 날짜 전체 환율 조회 (예: 2025-08-19일의 USD, JPY, EUR 환율 모두)
     */
    List<ExchangeRate> findAllByBaseDate(LocalDate baseDate);

    /**
     * 특정 통화코드의 최근 N일치 환율 조회
     */
    List<ExchangeRate> findAllByCurrencyCodeOrderByBaseDateDesc(String currencyCode);

    /**
     * 가장 최근 고시일자의 환율 조회
     */
    Optional<ExchangeRate> findFirstByCurrencyCodeOrderByBaseDateDesc(String currencyCode);
    /**
     * 특정 통화 코드와 날짜 범위 내의 환율 조회
     */
    List<ExchangeRate> findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
            String currencyCode, LocalDate startDate, LocalDate endDate
    );
}

