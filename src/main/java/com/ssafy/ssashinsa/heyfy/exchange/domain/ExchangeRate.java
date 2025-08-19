package com.ssafy.ssashinsa.heyfy.exchange.domain;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "exchange_rate")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRate {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "exchange_rate_id")
    private Long id;

    @Column(name = "base_date", nullable = false)
    private LocalDate baseDate;  // 고시 기준일자

    @Column(name = "currency_code", length = 3, nullable = false)
    private String currencyCode; // ISO 4217 코드 (USD, JPY 등)

    @Column(name = "country", length = 50)
    private String country;      // 선택 컬럼 (시각화, 검색용)

    @Column(name = "rate", precision = 15, scale = 6, nullable = false)
    private BigDecimal rate;     // 환율 값

    @Column(name = "unit", nullable = false)
    private Integer unit;        // 단위 (예: USD=1, JPY=100)

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}