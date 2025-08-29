package com.ssafy.ssashinsa.heyfy.exchange.domain;

import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@Builder
@Entity
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

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private Users user;

    public static ExchangeReservation create(Users user,
                                             String withdrawalAccountNo, Currency withdrawalAccountCurrency,
                                             String depositAccountNo, Currency depositAccountCurrency, Double transactionBalance) {

        return ExchangeReservation.builder()
                .user(user)
                .depositAccountNo(depositAccountNo)
                .depositAccountCurrency(depositAccountCurrency)
                .withdrawalAccountNo(withdrawalAccountNo)
                .withdrawalAccountCurrency(withdrawalAccountCurrency)
                .transactionBalance(transactionBalance)
                .exchangeCompleted(false)
                .build();
    }
}
