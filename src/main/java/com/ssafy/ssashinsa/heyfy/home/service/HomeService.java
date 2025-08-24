package com.ssafy.ssashinsa.heyfy.home.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CommonErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.home.dto.HomeDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@Slf4j
@RequiredArgsConstructor
public class HomeService {
    private final InquireService inquireService;

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository ForeignAccountRepository;

    public HomeDto getHome() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(CommonErrorCode.USER_NOT_FOUND));


        Optional<Account> optionalAccount = accountRepository.findByUser(user);
        Optional<ForeignAccount> optionalForeignAccount = ForeignAccountRepository.findByUser(user);

        HomeDto.HomeDtoBuilder homeDtoBuilder = HomeDto.builder()
                .studentId(studentId);


        optionalAccount.ifPresent(account -> {
            String accountNo = account.getAccountNo();


            ShinhanInquireSingleDepositResponseDto inquireResponse = inquireService.inquireSingleDeposit(accountNo);


            String currency = inquireResponse.getREC().getCurrency();
            double balance = inquireResponse.getREC().getAccountBalance();
            String accountName = inquireResponse.getREC().getAccountName();
            String bankName = inquireResponse.getREC().getBankName();

            homeDtoBuilder.normalAccount(
                    HomeDto.AccountInfo.builder()
                            .accountNo(accountNo)
                            .accountName(accountName)
                            .bankName(bankName)
                            .balance(String.valueOf((int) balance))
                            .currency(currency)
                            .build()
            );

        });

        optionalForeignAccount.ifPresent(foreignAccount -> {
            String accountNo = foreignAccount.getAccountNo();


            ShinhanInquireSingleDepositResponseDto inquireResponse = inquireService.inquireSingleForeignDeposit(accountNo);


            String currency = inquireResponse.getREC().getCurrency();
            double balance = inquireResponse.getREC().getAccountBalance();
            String accountName = inquireResponse.getREC().getAccountName();
            String bankName = inquireResponse.getREC().getBankName();

            homeDtoBuilder.foreignAccount(
                    HomeDto.AccountInfo.builder()
                            .accountNo(accountNo)
                            .accountName(accountName)
                            .bankName(bankName)
                            .balance(String.format("%.2f", balance))
                            .currency(currency)
                            .build()
            );
        });



        return homeDtoBuilder.build();
    }
}
