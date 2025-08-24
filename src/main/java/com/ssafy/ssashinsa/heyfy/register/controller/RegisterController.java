package com.ssafy.ssashinsa.heyfy.register.controller;

import com.ssafy.ssashinsa.heyfy.register.docs.CreateDepositAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.docs.CreateForeignDepositAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.register.service.RegisterService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/register")
@RequiredArgsConstructor
public class RegisterController {

    private final RegisterService registerService;


    @CreateDepositAccountDocs
    @PostMapping("/createdeposit")
    public ResponseEntity<AccountCreationResponseDto> createDepositAccount() {

        ShinhanCreateDepositResponseDto response = registerService.createDepositAccount();

        String message = response.getHeader().getResponseMessage();
        String accountNo = response.getREC().getAccountNo();
        String currency = response.getREC().getCurrency().getCurrency();

        AccountCreationResponseDto simplifiedResponse = new AccountCreationResponseDto(message, accountNo, currency);

        return ResponseEntity.ok(simplifiedResponse);
    }

    @CreateForeignDepositAccountDocs
    @PostMapping("/createforeigndeposit")
    public ResponseEntity<AccountCreationResponseDto> createDepositForeignAccount() {

        ShinhanCreateDepositResponseDto response = registerService.createForeignDepositAccount();

        String message = response.getHeader().getResponseMessage();
        String accountNo = response.getREC().getAccountNo();
        String currency = response.getREC().getCurrency().getCurrency();

        AccountCreationResponseDto simplifiedResponse = new AccountCreationResponseDto(message, accountNo, currency);

        return ResponseEntity.ok(simplifiedResponse);
    }
}
