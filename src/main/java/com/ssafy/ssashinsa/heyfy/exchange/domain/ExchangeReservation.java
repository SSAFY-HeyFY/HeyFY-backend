package com.ssafy.ssashinsa.heyfy.exchange.domain;

import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Getter
@Builder
@Entity
@EntityListeners(AuditingEntityListener.class)
@Table(name = "exchange_reservation")
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeReservation {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "exchange_reservation_id")
    private Long id;

    private String depositAccountNo;
    @Enumerated(EnumType.STRING)
    private Currency depositAccountCurrency;
    private String withdrawalAccountNo;
    @Enumerated(EnumType.STRING)
    private Currency withdrawalAccountCurrency;

    private Double transactionBalance;

    private boolean exchangeCompleted = false;
    private boolean isCanceled = false;

    private Double baseExchangeRate;

    @CreatedDate
    @Column(updatable = false, nullable = false)
    private LocalDateTime createdAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private Users user;

    public void cancel() {
        this.isCanceled = true;
    }


    public static ExchangeReservation create(Users user,
                                             String withdrawalAccountNo, Currency withdrawalAccountCurrency,
                                             String depositAccountNo, Currency depositAccountCurrency,
                                             Double transactionBalance, Double baseExchangeRate) {

        return ExchangeReservation.builder()
                .user(user)
                .depositAccountNo(depositAccountNo)
                .depositAccountCurrency(depositAccountCurrency)
                .withdrawalAccountNo(withdrawalAccountNo)
                .withdrawalAccountCurrency(withdrawalAccountCurrency)
                .transactionBalance(transactionBalance)
                .baseExchangeRate(baseExchangeRate)
                .exchangeCompleted(false)
                .build();
    }
}
